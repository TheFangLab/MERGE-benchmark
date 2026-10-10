#!/usr/bin/env python3
"""Merge per-GPU Carbon score shards in original input-row order."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--pattern", default="carbon_shard_*.csv")
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shard_dir = Path(args.shard_dir)
    paths = sorted(shard_dir.glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"no shard files matched {shard_dir / args.pattern}")

    frames = [pd.read_csv(path) for path in paths]
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values("input_row").reset_index(drop=True)
    output_cols = [
        "chrom",
        "pos",
        "ref",
        "alt",
        "carbon_score",
        "carbon_score_mean_delta",
        "carbon_ref_logp_sum",
        "carbon_alt_logp_sum",
        "carbon_ref_logp_mean",
        "carbon_alt_logp_mean",
        "carbon_token_count",
        "status",
        "error",
        "carbon_chrom_key",
        "carbon_window_start",
        "carbon_window_end",
        "input_row",
    ]
    cols = [col for col in output_cols if col in merged.columns]
    merged[cols].to_csv(args.output_csv, index=False)
    print(f"merged {len(paths)} shard files, {len(merged)} rows -> {args.output_csv}")


if __name__ == "__main__":
    main()
