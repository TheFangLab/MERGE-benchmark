#!/usr/bin/env python3
"""Evo2 variant scoring with explicit VEP-style score fields.

This script replaces an older evo2.py workflow.  The old file name
shadows the installed `evo2` package, and it only wrote the mean PLL difference,
which is too easy to confuse with paper/benchmark score columns.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pyfaidx import Fasta
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm


BASE = Path.home()
PUBLIC_CODING = BASE / "MERGE-paper" / "experiments" / "reproduce_evo2_clinvar_coding" / "hf_clinvar_vep_coding.csv"


def unshadow_evo2_package() -> None:
    """Avoid importing a stray evo2.py in the home directory instead of the installed package."""
    base = BASE.resolve()
    cleaned = []
    for item in sys.path:
        resolved = Path(item or os.getcwd()).resolve()
        if resolved == base:
            continue
        cleaned.append(item)
    sys.path[:] = cleaned


def normalize_chrom(value: object) -> str:
    text = str(value).strip()
    if text.startswith("chr"):
        text = text[3:]
    return text.upper()


def normalize_allele(value: object) -> str:
    return str(value).strip().upper()


def variant_key(df: pd.DataFrame, chrom_col: str, pos_col: str, ref_col: str, alt_col: str) -> pd.Series:
    return (
        df[chrom_col].map(normalize_chrom)
        + ":"
        + df[pos_col].astype(str)
        + ":"
        + df[ref_col].map(normalize_allele)
        + ":"
        + df[alt_col].map(normalize_allele)
    )


def get_fasta_chrom(genome: Fasta, chrom: object) -> str | None:
    chrom_norm = normalize_chrom(chrom)
    candidates = [
        chrom_norm,
        f"chr{chrom_norm}",
        f"NC_{int(chrom_norm):06d}.1" if chrom_norm.isdigit() else "",
        f"NC_{int(chrom_norm):06d}" if chrom_norm.isdigit() else "",
    ]
    if chrom_norm == "X":
        candidates.extend(["NC_000023.11", "NC_000023"])
    elif chrom_norm == "Y":
        candidates.extend(["NC_000024.10", "NC_000024"])
    for candidate in candidates:
        if candidate and candidate in genome.keys():
            return candidate
    return None


def extract_snv_context(
    genome: Fasta,
    chrom: object,
    pos: object,
    ref: object,
    alt: object,
    window_bp: int,
) -> tuple[str, str] | None:
    ref = normalize_allele(ref)
    alt = normalize_allele(alt)
    if len(ref) != 1 or len(alt) != 1 or ref not in "ACGT" or alt not in "ACGT":
        return None
    chrom_key = get_fasta_chrom(genome, chrom)
    if chrom_key is None:
        return None

    pos0 = int(pos) - 1
    chrom_len = len(genome[chrom_key])
    half = window_bp // 2
    start0 = max(0, pos0 - half)
    end0 = min(chrom_len, start0 + window_bp)
    start0 = max(0, end0 - window_bp)
    offset = pos0 - start0
    if offset < 0 or offset >= end0 - start0:
        return None

    ref_seq = str(genome[chrom_key][start0:end0]).upper()
    if len(ref_seq) < 2 or any(base not in "ACGT" for base in ref_seq):
        return None
    if ref_seq[offset] != ref:
        return None
    alt_seq = ref_seq[:offset] + alt + ref_seq[offset + 1 :]
    return ref_seq, alt_seq


def load_model(model_name: str, local_path: str | None):
    unshadow_evo2_package()
    from evo2.models import Evo2

    if torch.cuda.is_available():
        torch.set_default_dtype(torch.bfloat16)
    try:
        model = Evo2(model_name=model_name, local_path=local_path)
    finally:
        torch.set_default_dtype(torch.float32)
    return model


def batched_score(model, seqs: list[str], batch_size: int, reduce_method: str, average_rc: bool) -> np.ndarray:
    scores = model.score_sequences(
        seqs,
        batch_size=batch_size,
        reduce_method=reduce_method,
        average_reverse_complement=average_rc,
    )
    return np.asarray(scores, dtype=float)


def compute_metrics(y: pd.Series, score: pd.Series) -> dict[str, float | str | int]:
    valid = y.notna() & score.notna()
    yv = y.loc[valid].astype(int)
    sv = score.loc[valid].astype(float)
    if len(yv) == 0 or yv.nunique() < 2:
        return {"n": int(valid.sum())}
    raw_auc = roc_auc_score(yv, sv)
    flip_auc = roc_auc_score(yv, -sv)
    raw_ap = average_precision_score(yv, sv)
    flip_ap = average_precision_score(yv, -sv)
    if raw_auc >= flip_auc:
        return {
            "n": int(valid.sum()),
            "auroc": float(raw_auc),
            "ap": float(raw_ap),
            "direction": "higher_pathogenic",
        }
    return {
        "n": int(valid.sum()),
        "auroc": float(flip_auc),
        "ap": float(flip_ap),
        "direction": "lower_pathogenic",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", default=str(PUBLIC_CODING))
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--model-name", default="evo2_7b")
    parser.add_argument("--local-model-path", default=None)
    parser.add_argument("--window-bp", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--end-row", type=int, default=None)
    parser.add_argument("--chrom-col", default=None)
    parser.add_argument("--pos-col", default="pos")
    parser.add_argument("--ref-col", default="ref")
    parser.add_argument("--alt-col", default="alt")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--public-score-col", default="Evo2-7B")
    parser.add_argument("--average-rc", action="store_true")
    parser.add_argument("--write-summary", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    if args.start_row or args.end_row is not None:
        df = df.iloc[args.start_row:args.end_row].copy()
    if args.limit:
        df = df.head(args.limit).copy()
    chrom_col = args.chrom_col or ("chrom" if "chrom" in df.columns else "chr")

    genome = Fasta(args.fasta, sequence_always_upper=True, as_raw=True)
    ref_seqs: list[str] = []
    alt_seqs: list[str] = []
    valid_indices: list[int] = []
    skipped = 0
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting SNV windows"):
        pair = extract_snv_context(
            genome,
            row[chrom_col],
            row[args.pos_col],
            row[args.ref_col],
            row[args.alt_col],
            args.window_bp,
        )
        if pair is None:
            skipped += 1
            continue
        ref_seq, alt_seq = pair
        ref_seqs.append(ref_seq)
        alt_seqs.append(alt_seq)
        valid_indices.append(idx)

    if not valid_indices:
        raise RuntimeError("No valid SNV windows were extracted. Check genome build, chromosome names, and ref alleles.")

    model = load_model(args.model_name, args.local_model_path)

    rows = df.loc[valid_indices].reset_index(drop=True).copy()
    all_seqs = ref_seqs + alt_seqs

    ref_mean, alt_mean = np.split(
        batched_score(model, all_seqs, args.batch_size, "mean", args.average_rc),
        2,
    )
    # Evo2's package mean reduction averages next-token logprobs. For a sequence
    # of length L, there are L-1 predicted tokens, so the corresponding PLL sum
    # is mean * (L-1). Avoiding a second forward pass cuts runtime roughly in half.
    token_counts = np.asarray([max(len(seq) - 1, 0) for seq in ref_seqs], dtype=float)
    ref_sum = ref_mean * token_counts
    alt_sum = alt_mean * token_counts

    rows["evo2_ref_mean_pll"] = ref_mean
    rows["evo2_alt_mean_pll"] = alt_mean
    rows["evo2_ref_sum_pll"] = ref_sum
    rows["evo2_alt_sum_pll"] = alt_sum
    rows["evo2_ref_minus_alt_mean"] = ref_mean - alt_mean
    rows["evo2_alt_minus_ref_mean"] = alt_mean - ref_mean
    rows["evo2_ref_minus_alt_sum"] = ref_sum - alt_sum
    rows["evo2_alt_minus_ref_sum"] = alt_sum - ref_sum
    # Official BRCA1 notebook: delta_scores = var_scores - ref_scores using
    # model.score_sequences() default mean PLL. Lower delta is more disruptive,
    # so -delta is the pathogenicity-oriented score used for AUROC.
    rows["evo2_delta_score_brca1_official"] = rows["evo2_alt_minus_ref_mean"]
    rows["evo2_pathogenic_score_brca1_official"] = -rows["evo2_delta_score_brca1_official"]
    rows["evo2_score"] = rows["evo2_pathogenic_score_brca1_official"]
    rows["evo2_score_definition"] = "brca1_official_negative_alt_mean_minus_ref_mean"
    rows["evo2_context_length"] = [len(seq) for seq in ref_seqs]
    rows["evo2_token_count"] = token_counts.astype(int)
    rows["evo2_average_reverse_complement"] = bool(args.average_rc)

    rows.to_csv(out_csv, index=False)

    summary: dict[str, object] = {
        "input_csv": str(input_csv),
        "output_csv": str(out_csv),
        "n_input": int(len(df)),
        "n_scored": int(len(rows)),
        "n_skipped": int(skipped),
        "window_bp": int(args.window_bp),
        "model_name": args.model_name,
        "score_primary": "evo2_pathogenic_score_brca1_official",
        "score_note": "Official BRCA1 notebook computes delta = alt_mean_pll - ref_mean_pll; lower delta is more disruptive, so this script uses -delta as evo2_score.",
    }

    if args.label_col in rows.columns:
        metric_rows = {}
        for col in [
            "evo2_pathogenic_score_brca1_official",
            "evo2_delta_score_brca1_official",
            "evo2_ref_minus_alt_sum",
            "evo2_ref_minus_alt_mean",
            "evo2_alt_minus_ref_sum",
            "evo2_alt_minus_ref_mean",
        ]:
            metric_rows[col] = compute_metrics(rows[args.label_col], rows[col])
        summary["label_metrics"] = metric_rows

    if args.public_score_col in rows.columns:
        corr_rows = {}
        for col in ["evo2_pathogenic_score_brca1_official", "evo2_ref_minus_alt_sum", "evo2_ref_minus_alt_mean"]:
            valid = rows[col].notna() & rows[args.public_score_col].notna()
            if valid.sum() > 2:
                corr_rows[col] = float(spearmanr(rows.loc[valid, col], rows.loc[valid, args.public_score_col]).statistic)
        summary["spearman_vs_public_score"] = corr_rows

    summary_path = Path(args.write_summary) if args.write_summary else out_csv.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
