#!/usr/bin/env python3
"""Precompute Carbon-3B SNV effect scores for MERGE feature matrices."""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pyfaidx import Fasta
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_ID = "HuggingFaceBio/Carbon-3B"


@dataclass(frozen=True)
class SequencePair:
    ref_sequence: str
    alt_sequence: str
    chrom_key: str
    window_start_1based: int
    window_end_1based: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score SNV ref/alt sequence windows with Carbon-3B for MERGE."
    )
    parser.add_argument("--input-csv", required=True, help="Input CSV with chrom/pos/ref/alt columns.")
    parser.add_argument("--output-csv", required=True, help="Output CSV path.")
    parser.add_argument("--fasta", required=True, help="hg38/GRCh38 reference FASTA path.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="HF model id or local model path.")
    parser.add_argument("--revision", default=None, help="Optional model revision, e.g. fns.")
    parser.add_argument("--chrom-col", default="chrom", help="Chromosome column name.")
    parser.add_argument("--pos-col", default="pos", help="1-based position column name.")
    parser.add_argument("--ref-col", default="ref", help="Reference allele column name.")
    parser.add_argument("--alt-col", default="alt", help="Alternate allele column name.")
    parser.add_argument("--window-bp", type=int, default=24576, help="Sequence window length in bp. Must be divisible by 6.")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of variants per GPU batch.")
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0, cpu, etc.")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    parser.add_argument("--resume", action="store_true", help="Append after existing rows in output CSV.")
    parser.add_argument("--shard-index", type=int, default=0, help="0-based shard index.")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards.")
    parser.add_argument("--keep-sequences", action="store_true", help="Include ref_sequence and alt_sequence in output.")
    parser.add_argument("--skip-non-acgt", action="store_true", default=True)
    parser.add_argument("--no-skip-non-acgt", dest="skip_non_acgt", action="store_false")
    return parser.parse_args()


def normalize_chrom(raw: object) -> list[str]:
    value = str(raw).strip()
    if value.lower().startswith("chr"):
        bare = value[3:]
        return [value, bare]
    return [f"chr{value}", value]


def build_chrom_aliases(genome: Fasta) -> dict[str, str]:
    aliases: dict[str, str] = {}
    refseq = {
        "NC_000001": "1", "NC_000002": "2", "NC_000003": "3", "NC_000004": "4",
        "NC_000005": "5", "NC_000006": "6", "NC_000007": "7", "NC_000008": "8",
        "NC_000009": "9", "NC_000010": "10", "NC_000011": "11", "NC_000012": "12",
        "NC_000013": "13", "NC_000014": "14", "NC_000015": "15", "NC_000016": "16",
        "NC_000017": "17", "NC_000018": "18", "NC_000019": "19", "NC_000020": "20",
        "NC_000021": "21", "NC_000022": "22", "NC_000023": "X", "NC_000024": "Y",
    }
    for key in genome.keys():
        primary = key.split()[0]
        stem = primary.split(".")[0]
        candidates = {key, primary, stem}
        if stem in refseq:
            bare = refseq[stem]
            candidates.update({bare, f"chr{bare}"})
        if primary.lower().startswith("chr"):
            candidates.add(primary[3:])
        else:
            candidates.add(f"chr{primary}")
        for name in candidates:
            aliases[name] = key
    return aliases


def resolve_chrom(chrom: object, aliases: dict[str, str]) -> str:
    for candidate in normalize_chrom(chrom):
        if candidate in aliases:
            return aliases[candidate]
    raise ValueError(f"chromosome not found in FASTA: {chrom}")


def make_snv_pair(
    genome: Fasta,
    aliases: dict[str, str],
    chrom: object,
    pos: object,
    ref: object,
    alt: object,
    window_bp: int,
    skip_non_acgt: bool,
) -> SequencePair:
    if window_bp % 6 != 0:
        raise ValueError("--window-bp must be divisible by 6 for Carbon's 6-mer tokenizer")

    ref = str(ref).upper().strip()
    alt = str(alt).upper().strip()
    if len(ref) != 1 or len(alt) != 1:
        raise ValueError("only SNVs are supported by this script")
    if skip_non_acgt and (ref not in "ACGT" or alt not in "ACGT"):
        raise ValueError("non-ACGT ref/alt allele")

    chrom_key = resolve_chrom(chrom, aliases)
    pos_1based = int(pos)
    chrom_len = len(genome[chrom_key])
    if pos_1based < 1 or pos_1based > chrom_len:
        raise ValueError(f"position out of range: {chrom}:{pos}")

    half_left = window_bp // 2
    start0 = pos_1based - 1 - half_left
    end0 = start0 + window_bp
    if start0 < 0:
        end0 -= start0
        start0 = 0
    if end0 > chrom_len:
        start0 = max(0, start0 - (end0 - chrom_len))
        end0 = chrom_len

    length = end0 - start0
    length -= length % 6
    if length <= 0:
        raise ValueError("empty sequence after 6-mer truncation")
    end0 = start0 + length
    if not (start0 <= pos_1based - 1 < end0):
        raise ValueError("variant fell outside truncated sequence window")

    ref_sequence = str(genome[chrom_key][start0:end0]).upper()
    snv_index = pos_1based - 1 - start0
    observed = ref_sequence[snv_index]
    if observed != ref:
        raise ValueError(f"reference mismatch: FASTA has {observed}, input has {ref}")
    if skip_non_acgt and any(base not in "ACGT" for base in ref_sequence):
        raise ValueError("sequence window contains non-ACGT base")

    alt_sequence = ref_sequence[:snv_index] + alt + ref_sequence[snv_index + 1 :]
    return SequencePair(
        ref_sequence=ref_sequence,
        alt_sequence=alt_sequence,
        chrom_key=chrom_key,
        window_start_1based=start0 + 1,
        window_end_1based=end0,
    )


def torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def load_model(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.revision,
        torch_dtype=torch_dtype(args.dtype),
        trust_remote_code=args.trust_remote_code if args.revision == "fns" else False,
        low_cpu_mem_usage=True,
    )
    model.to(args.device)
    model.eval()
    return tokenizer, model


@torch.inference_mode()
def score_sequences(
    sequences: list[str],
    tokenizer,
    model,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    texts = [f"<dna>{seq}</dna>" for seq in sequences]
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        add_special_tokens=False,
        padding=True,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    target_mask = attention_mask[:, 1:].bool()
    token_logp = F.log_softmax(logits.float(), dim=-1).gather(
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)
    token_logp = token_logp.masked_fill(~target_mask, 0.0)

    total_logp = token_logp.sum(dim=1)
    token_count = target_mask.sum(dim=1).clamp_min(1)
    mean_logp = total_logp / token_count
    return (
        total_logp.detach().cpu().numpy(),
        mean_logp.detach().cpu().numpy(),
        token_count.detach().cpu().numpy(),
    )


def append_rows(path: Path, rows: list[dict], header: bool) -> None:
    pd.DataFrame(rows).to_csv(path, mode="a", header=header, index=False)


def existing_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    return max(0, sum(1 for _ in path.open("r", encoding="utf-8")) - 1)


def batched(items: list[tuple[int, pd.Series]], batch_size: int) -> Iterable[list[tuple[int, pd.Series]]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Run this on gn1 or set --device cpu.")

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    required = [args.chrom_col, args.pos_col, args.ref_col, args.alt_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard-index must be in [0, --num-shards)")
    shard_mask = np.arange(len(df)) % args.num_shards == args.shard_index
    df = df.loc[shard_mask].reset_index(drop=False).rename(columns={"index": "input_row"})

    start_at = existing_row_count(output_path) if args.resume else 0
    if start_at >= len(df):
        print(f"Nothing to do: {output_path} already has {start_at} shard rows.")
        return
    if not args.resume and output_path.exists():
        output_path.unlink()

    print(f"Loading FASTA: {args.fasta}")
    genome = Fasta(args.fasta, sequence_always_upper=True)
    aliases = build_chrom_aliases(genome)

    print(f"Loading Carbon model: {args.model_id}")
    tokenizer, model = load_model(args)

    work_items = list(df.iloc[start_at:].iterrows())
    wrote_header = output_path.exists() and output_path.stat().st_size > 0

    for batch in tqdm(list(batched(work_items, args.batch_size)), desc=f"carbon shard {args.shard_index}"):
        pairs: list[SequencePair] = []
        rows: list[dict] = []

        for _, row in batch:
            base = {
                "input_row": int(row["input_row"]),
                "chrom": row[args.chrom_col],
                "pos": int(row[args.pos_col]),
                "ref": str(row[args.ref_col]).upper(),
                "alt": str(row[args.alt_col]).upper(),
            }
            try:
                pair = make_snv_pair(
                    genome=genome,
                    aliases=aliases,
                    chrom=row[args.chrom_col],
                    pos=row[args.pos_col],
                    ref=row[args.ref_col],
                    alt=row[args.alt_col],
                    window_bp=args.window_bp,
                    skip_non_acgt=args.skip_non_acgt,
                )
                pairs.append(pair)
                base.update(
                    {
                        "status": "ok",
                        "error": "",
                        "carbon_chrom_key": pair.chrom_key,
                        "carbon_window_start": pair.window_start_1based,
                        "carbon_window_end": pair.window_end_1based,
                    }
                )
            except Exception as exc:
                base.update(
                    {
                        "status": "error",
                        "error": str(exc),
                        "carbon_chrom_key": "",
                        "carbon_window_start": np.nan,
                        "carbon_window_end": np.nan,
                        "carbon_ref_logp_sum": np.nan,
                        "carbon_alt_logp_sum": np.nan,
                        "carbon_ref_logp_mean": np.nan,
                        "carbon_alt_logp_mean": np.nan,
                        "carbon_token_count": np.nan,
                        "carbon_score": np.nan,
                        "carbon_score_mean_delta": np.nan,
                    }
                )
            rows.append(base)

        valid_indices = [i for i, row in enumerate(rows) if row["status"] == "ok"]
        if valid_indices:
            seqs: list[str] = []
            for pair in pairs:
                seqs.extend([pair.ref_sequence, pair.alt_sequence])

            totals, means, token_counts = score_sequences(seqs, tokenizer, model, args.device)
            for local_i, pair_i in enumerate(valid_indices):
                ref_i = 2 * local_i
                alt_i = ref_i + 1
                rows[pair_i].update(
                    {
                        "carbon_ref_logp_sum": float(totals[ref_i]),
                        "carbon_alt_logp_sum": float(totals[alt_i]),
                        "carbon_ref_logp_mean": float(means[ref_i]),
                        "carbon_alt_logp_mean": float(means[alt_i]),
                        "carbon_token_count": int(token_counts[ref_i]),
                        "carbon_score": float(totals[ref_i] - totals[alt_i]),
                        "carbon_score_mean_delta": float(means[ref_i] - means[alt_i]),
                    }
                )
                if args.keep_sequences:
                    rows[pair_i]["ref_sequence"] = pairs[local_i].ref_sequence
                    rows[pair_i]["alt_sequence"] = pairs[local_i].alt_sequence

        append_rows(output_path, rows, header=not wrote_header)
        wrote_header = True

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Saved Carbon scores to {output_path}")


if __name__ == "__main__":
    main()
