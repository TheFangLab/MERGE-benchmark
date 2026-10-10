#!/usr/bin/env python3
"""Extract causal-LM Non-SNV features from Evo2 or Carbon."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pyfaidx import Fasta
from tqdm import tqdm

from non_snv_sequences import build_chrom_aliases, make_sequence_pair


# Official Evo 2 weights (<EVO2_WEIGHTS_DIR>/<name>/<name>.pt) and the Carbon-3B
# Hugging Face snapshot fe755cb5c7498acbf630080609ef61ecc4e36c17.
_EVO2_DIR = os.environ.get("EVO2_WEIGHTS_DIR", "models")
EVO2_PATHS = {name: os.path.join(_EVO2_DIR, name, f"{name}.pt")
              for name in ("evo2_1b_base", "evo2_7b_base", "evo2_7b")}
CARBON_PATH = os.environ.get("CARBON_MODEL_PATH", "HuggingFaceBio/Carbon-3B")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["evo2", "carbon"], required=True)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--fasta", default=os.environ.get("HG38_FASTA", "hg38.fa"))
    parser.add_argument("--window-bp", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--end-row", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def completed_rows(path: Path) -> set[int]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    return set(pd.read_csv(path, usecols=["input_row"])["input_row"].astype(int))


def append_records(path: Path, records: list[dict[str, object]]) -> None:
    pd.DataFrame(records).to_csv(
        path,
        mode="a",
        header=not path.exists() or path.stat().st_size == 0,
        index=False,
    )


def load_evo2(model_name: str):
    if model_name not in EVO2_PATHS:
        raise ValueError(f"model-name must be one of {sorted(EVO2_PATHS)}")
    if os.environ.get("EVO2_SRC"):               # a source checkout of the evo2 package
        sys.path.insert(0, os.environ["EVO2_SRC"])
    from evo2.models import Evo2

    torch.set_default_dtype(torch.bfloat16)
    try:
        model = Evo2(model_name=model_name, local_path=EVO2_PATHS[model_name])
    finally:
        torch.set_default_dtype(torch.float32)
    return model


def score_evo2(model, refs: list[str], alts: list[str], batch_size: int):
    sequences = refs + alts
    means = np.asarray(
        model.score_sequences(
            sequences,
            batch_size=batch_size,
            reduce_method="mean",
            average_reverse_complement=False,
        ),
        dtype=float,
    )
    ref_mean, alt_mean = np.split(means, 2)
    ref_tokens = np.asarray([max(len(seq) - 1, 1) for seq in refs])
    alt_tokens = np.asarray([max(len(seq) - 1, 1) for seq in alts])
    return (
        ref_mean,
        alt_mean,
        ref_mean * ref_tokens,
        alt_mean * alt_tokens,
        ref_tokens,
        alt_tokens,
    )


def load_carbon(device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        CARBON_PATH,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        CARBON_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(device)
    model.eval()
    return tokenizer, model


@torch.inference_mode()
def score_carbon(tokenizer, model, refs: list[str], alts: list[str], device: str):
    texts = [f"<dna>{seq}</dna>" for seq in refs + alts]
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        add_special_tokens=False,
        padding=True,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).logits[:, :-1, :]
    targets = input_ids[:, 1:]
    mask = attention_mask[:, 1:].bool()
    token_logp = F.log_softmax(logits.float(), dim=-1).gather(
        -1, targets.unsqueeze(-1)
    ).squeeze(-1)
    sums = token_logp.masked_fill(~mask, 0).sum(1)
    counts = mask.sum(1).clamp_min(1)
    means = sums / counts
    n = len(refs)
    return (
        means[:n].cpu().numpy(),
        means[n:].cpu().numpy(),
        sums[:n].cpu().numpy(),
        sums[n:].cpu().numpy(),
        counts[:n].cpu().numpy(),
        counts[n:].cpu().numpy(),
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required")
    model_name = args.model_name or (
        "evo2_7b" if args.backend == "evo2" else "carbon_3b"
    )
    window_bp = args.window_bp or (8192 if args.backend == "evo2" else 24576)
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.resume:
        output.unlink()
    done = completed_rows(output) if args.resume else set()

    data = pd.read_csv(args.input_csv)
    stop = len(data) if args.end_row is None else min(args.end_row, len(data))
    data = data.iloc[args.start_row:stop]
    genome_fasta = Fasta(args.fasta, sequence_always_upper=True, as_raw=True)
    aliases = build_chrom_aliases(genome_fasta)

    if args.backend == "evo2":
        model = load_evo2(model_name)
        tokenizer = None
    else:
        tokenizer, model = load_carbon(args.device)

    pending: list[tuple[int, pd.Series, object]] = []
    for input_row, row in tqdm(data.iterrows(), total=len(data), desc=model_name):
        if int(input_row) in done:
            continue
        try:
            pair = make_sequence_pair(
                genome_fasta,
                aliases,
                row["chr"],
                row["pos"],
                row["ref"],
                row["alt"],
                window_bp,
            )
            pending.append((int(input_row), row, pair))
        except Exception as exc:
            append_records(
                output,
                [{
                    "input_row": int(input_row),
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "model_name": model_name,
                }],
            )
            continue

        if len(pending) < args.batch_size:
            continue
        write_batch(args, output, model_name, model, tokenizer, pending, window_bp)
        pending = []

    if pending:
        write_batch(args, output, model_name, model, tokenizer, pending, window_bp)


def write_batch(args, output, model_name, model, tokenizer, pending, window_bp):
    refs = [item[2].ref_sequence for item in pending]
    alts = [item[2].alt_sequence for item in pending]
    if args.backend == "evo2":
        values = score_evo2(model, refs, alts, args.batch_size)
    else:
        values = score_carbon(tokenizer, model, refs, alts, args.device)
    ref_mean, alt_mean, ref_sum, alt_sum, ref_tokens, alt_tokens = values
    records = []
    for i, (input_row, row, pair) in enumerate(pending):
        prefix = model_name.lower().replace("-", "_")
        record = {
            "input_row": input_row,
            "chr": row["chr"],
            "pos": int(row["pos"]),
            "ref": row["ref"],
            "alt": row["alt"],
            "status": "ok",
            "error": "",
            "model_name": model_name,
            "window_bp": window_bp,
            "ref_sequence_length": len(pair.ref_sequence),
            "alt_sequence_length": len(pair.alt_sequence),
            f"{prefix}_ref_mean_logp": float(ref_mean[i]),
            f"{prefix}_alt_mean_logp": float(alt_mean[i]),
            f"{prefix}_ref_sum_logp": float(ref_sum[i]),
            f"{prefix}_alt_sum_logp": float(alt_sum[i]),
            f"{prefix}_ref_token_count": int(ref_tokens[i]),
            f"{prefix}_alt_token_count": int(alt_tokens[i]),
            f"{prefix}_score_mean": float(ref_mean[i] - alt_mean[i]),
            f"{prefix}_score_sum": float(ref_sum[i] - alt_sum[i]),
        }
        records.append(record)
    append_records(output, records)
    print(json.dumps({"model": model_name, "last_row": records[-1]["input_row"]}), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
