#!/usr/bin/env python3
"""Extract compact AlphaGenome features for ClinVar Non-SNV variants."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from alphagenome.data import genome
from alphagenome.models import dna_client, variant_scorers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument(
        "--api-key-file",
        default=str(Path.home() / ".config/alphagenome/api_key"),
    )
    parser.add_argument("--sequence-length", type=int, default=2**20)
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--end-row", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--scorers", nargs="*", default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--max-attempts", type=int, default=4)
    return parser.parse_args()


def chromosome_name(value: object) -> str:
    text = str(value).strip()
    return text if text.startswith("chr") else f"chr{text}"


def make_interval(variant: genome.Variant, width: int) -> genome.Interval:
    start = max(0, variant.start - width // 2)
    return genome.Interval(variant.chromosome, start, start + width)


def finite_summary(values: np.ndarray, prefix: str) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_mean": np.nan,
            f"{prefix}_mean_abs": np.nan,
            f"{prefix}_rms": np.nan,
            f"{prefix}_max_abs": np.nan,
            f"{prefix}_signed_max_abs": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
        }
    max_abs_index = int(np.argmax(np.abs(values)))
    return {
        f"{prefix}_n": int(values.size),
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_mean_abs": float(np.mean(np.abs(values))),
        f"{prefix}_rms": float(np.sqrt(np.mean(np.square(values)))),
        f"{prefix}_max_abs": float(np.max(np.abs(values))),
        f"{prefix}_signed_max_abs": float(values[max_abs_index]),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
    }


def scorer_features(scores) -> dict[str, float | int]:
    features: dict[str, float | int] = {}
    for adata in scores:
        scorer = adata.uns["variant_scorer"]
        name = next(
            (
                key.lower()
                for key, value in variant_scorers.RECOMMENDED_VARIANT_SCORERS.items()
                if value == scorer
            ),
            scorer.requested_output.name.lower(),
        )
        features.update(finite_summary(adata.X, f"alphagenome_{name}_raw"))
        if "quantiles" in adata.layers:
            features.update(
                finite_summary(
                    adata.layers["quantiles"],
                    f"alphagenome_{name}_quantile",
                )
            )
    return features


def existing_rows(path: Path) -> set[int]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    return set(pd.read_csv(path, usecols=["input_row"])["input_row"].astype(int))


def append_record(path: Path, record: dict[str, object]) -> None:
    frame = pd.DataFrame([record])
    frame.to_csv(
        path,
        mode="a",
        header=not path.exists() or path.stat().st_size == 0,
        index=False,
    )


def main() -> None:
    args = parse_args()
    dna_client.validate_sequence_length(args.sequence_length)
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.resume:
        output.unlink()

    api_key = Path(args.api_key_file).read_text(encoding="utf-8").strip()
    if not api_key:
        raise RuntimeError("AlphaGenome API key file is empty")
    model = dna_client.create(api_key, timeout=30)

    scorer_map = variant_scorers.RECOMMENDED_VARIANT_SCORERS
    scorer_names = args.scorers or list(scorer_map)
    unknown = sorted(set(scorer_names) - set(scorer_map))
    if unknown:
        raise ValueError(f"unknown scorer names: {unknown}")
    scorers = [scorer_map[name] for name in scorer_names]

    data = pd.read_csv(args.input_csv)
    stop = len(data) if args.end_row is None else min(args.end_row, len(data))
    selected = data.iloc[args.start_row:stop]
    done = existing_rows(output) if args.resume else set()

    for input_row, row in selected.iterrows():
        if int(input_row) in done:
            continue
        record: dict[str, object] = {
            "input_row": int(input_row),
            "chr": row["chr"],
            "pos": int(row["pos"]),
            "ref": str(row["ref"]),
            "alt": str(row["alt"]),
            "status": "error",
            "error": "",
            "alphagenome_sequence_length": int(args.sequence_length),
            "alphagenome_scorers": ",".join(scorer_names),
        }
        started = time.time()
        try:
            variant = genome.Variant(
                chromosome=chromosome_name(row["chr"]),
                position=int(row["pos"]),
                reference_bases=str(row["ref"]).upper(),
                alternate_bases=str(row["alt"]).upper(),
            )
            interval = make_interval(variant, args.sequence_length)
            last_error: Exception | None = None
            for attempt in range(1, args.max_attempts + 1):
                try:
                    scores = model.score_variant(
                        interval=interval,
                        variant=variant,
                        variant_scorers=scorers,
                    )
                    last_error = None
                    break
                except Exception as exc:  # API errors are recorded and retried.
                    last_error = exc
                    if attempt < args.max_attempts:
                        time.sleep(min(60.0, 2.0**attempt))
            if last_error is not None:
                raise last_error
            record.update(scorer_features(scores))
            record["status"] = "ok"
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"[:1000]
        record["elapsed_seconds"] = round(time.time() - started, 3)
        append_record(output, record)
        print(json.dumps(record, default=str), flush=True)
        if args.sleep_seconds:
            time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    main()
