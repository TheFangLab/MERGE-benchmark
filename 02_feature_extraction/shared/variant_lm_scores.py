#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pyfaidx import Fasta
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer


def norm_chrom(value: object) -> str:
    text = str(value).strip()
    return text if text.startswith("chr") else f"chr{text}"


def norm_base(value: object) -> str:
    return str(value).strip().upper()


def chrom_key(genome: Fasta, chrom: object) -> str | None:
    direct = norm_chrom(chrom)
    bare = direct[3:] if direct.startswith("chr") else direct
    for candidate in (direct, bare):
        if candidate in genome.keys():
            return candidate
    return None


def extract_context(
    genome: Fasta,
    chrom: object,
    pos: object,
    ref: object,
    alt: object,
    window_bp: int,
    multiple_of: int = 1,
) -> tuple[str, str] | None:
    ref = norm_base(ref)
    alt = norm_base(alt)
    if len(ref) != 1 or len(alt) != 1 or ref not in "ACGT" or alt not in "ACGT":
        return None

    key = chrom_key(genome, chrom)
    if key is None:
        return None

    pos0 = int(pos) - 1
    half = window_bp // 2
    start = max(0, pos0 - half)
    end = min(len(genome[key]), start + window_bp)
    start = max(0, end - window_bp)
    offset = pos0 - start
    if offset < 0 or offset >= end - start:
        return None

    seq = str(genome[key][start:end]).upper()
    if multiple_of > 1:
        trim = len(seq) % multiple_of
        if trim:
            seq = seq[trim:]
            offset -= trim
    if offset < 0 or offset >= len(seq):
        return None
    if seq[offset] != ref or any(base not in "ACGT" for base in seq):
        return None
    return seq, seq[:offset] + alt + seq[offset + 1 :]


def mean_pool(last_hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.unsqueeze(-1).to(last_hidden.dtype)
    return (last_hidden * mask_f).sum(1) / mask_f.sum(1).clamp_min(1)


@torch.inference_mode()
def score_masked_lm(model, tokenizer, seqs_ref: list[str], seqs_alt: list[str], device: torch.device, max_length: int) -> np.ndarray:
    encoded = tokenizer(seqs_ref + seqs_alt, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    out = model(**encoded, output_hidden_states=True)
    emb = mean_pool(out.hidden_states[-1], encoded["attention_mask"])
    ref_emb, alt_emb = emb[: len(seqs_ref)], emb[len(seqs_ref) :]
    return torch.linalg.vector_norm(alt_emb - ref_emb, dim=1).float().cpu().numpy()


@torch.inference_mode()
def sequence_nll(model, tokenizer, seqs: list[str], device: torch.device, max_length: int) -> torch.Tensor:
    encoded = tokenizer(seqs, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[:, :-1, :].float()
    labels = input_ids[:, 1:]
    mask = attention_mask[:, 1:].bool()
    log_probs = torch.log_softmax(logits, dim=-1)
    token_ll = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    lengths = mask.sum(1).clamp_min(1)
    return -(token_ll.masked_fill(~mask, 0).sum(1) / lengths)


def score_causal_lm(model, tokenizer, seqs_ref: list[str], seqs_alt: list[str], device: torch.device, max_length: int) -> np.ndarray:
    ref_nll = sequence_nll(model, tokenizer, seqs_ref, device, max_length)
    alt_nll = sequence_nll(model, tokenizer, seqs_alt, device, max_length)
    return (alt_nll - ref_nll).float().cpu().numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--fasta", default=os.environ.get("HG38_FASTA", "hg38.fa"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-kind", choices=["causal", "masked"], required=True)
    parser.add_argument("--score-col", required=True)
    parser.add_argument("--window-bp", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--end-row", type=int, default=None)
    parser.add_argument("--chrom-col", default="chr")
    parser.add_argument("--pos-col", default="pos")
    parser.add_argument("--ref-col", default="ref")
    parser.add_argument("--alt-col", default="alt")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--multiple-of", type=int, default=1)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    if args.start_row or args.end_row is not None:
        df = df.iloc[args.start_row : args.end_row].copy()
    if args.limit:
        df = df.head(args.limit).copy()

    genome = Fasta(args.fasta, sequence_always_upper=True, as_raw=True)
    ref_seqs: list[str] = []
    alt_seqs: list[str] = []
    valid_indices: list[int] = []
    skipped = 0
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="extract"):
        pair = extract_context(
            genome,
            row[args.chrom_col],
            row[args.pos_col],
            row[args.ref_col],
            row[args.alt_col],
            args.window_bp,
            args.multiple_of,
        )
        if pair is None:
            skipped += 1
            continue
        ref, alt = pair
        ref_seqs.append(ref)
        alt_seqs.append(alt)
        valid_indices.append(idx)

    if not valid_indices:
        raise RuntimeError("no valid SNV windows extracted")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    max_length = args.max_length or min(args.window_bp, getattr(tokenizer, "model_max_length", args.window_bp))
    if max_length > 100000000:
        max_length = args.window_bp

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.bf16 and device.type == "cuda" else None
    model_cls = AutoModelForCausalLM if args.model_kind == "causal" else AutoModelForMaskedLM
    load_kwargs = {
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": dtype,
    }
    try:
        import accelerate  # noqa: F401

        load_kwargs["low_cpu_mem_usage"] = True
    except Exception:
        pass
    model = model_cls.from_pretrained(args.model, **load_kwargs).to(device)
    model.eval()

    scores: list[float] = []
    for start in tqdm(range(0, len(ref_seqs), args.batch_size), desc="score"):
        refs = ref_seqs[start : start + args.batch_size]
        alts = alt_seqs[start : start + args.batch_size]
        if args.model_kind == "causal":
            batch_scores = score_causal_lm(model, tokenizer, refs, alts, device, max_length)
        else:
            batch_scores = score_masked_lm(model, tokenizer, refs, alts, device, max_length)
        scores.extend(float(x) for x in batch_scores)

    out = df.loc[valid_indices].copy()
    out[args.score_col] = scores
    out[f"{args.score_col}_model"] = args.model
    out[f"{args.score_col}_window_bp"] = args.window_bp
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)

    summary = {
        "input_csv": args.input_csv,
        "output_csv": args.output_csv,
        "model": args.model,
        "model_kind": args.model_kind,
        "score_col": args.score_col,
        "n_input": int(len(df)),
        "n_scored": int(len(out)),
        "n_skipped": int(skipped),
        "window_bp": int(args.window_bp),
        "max_length": int(max_length),
    }
    summary_path = Path(args.summary_json) if args.summary_json else Path(args.output_csv).with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
