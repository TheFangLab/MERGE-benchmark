#!/usr/bin/env python3
"""alphagenome_splicing: AlphaGenome's splicing score for SNVs.

  alphagenome_splicing = max(SPLICE_SITES) + max(SPLICE_SITE_USAGE) + max(SPLICE_JUNCTIONS) / 5

Each term is the maximum raw score over all tracks of that output type, from the
three splicing entries of RECOMMENDED_VARIANT_SCORERS, on a 1 Mb interval
(variant.reference_interval.resize(SEQUENCE_LENGTH_1MB)) for Homo sapiens. An output
type that returns no score contributes 0; the division by 5 applies to the junction
term only. This is the computation used for the splicing benchmark table and the
splice-region precomputed scores.

The six global alphagenome_* columns come from alphagenome_non_snv.py and
summarize_alphagenome.py instead. Those use a differently centred interval and all
scorers, so their per-scorer maxima do not reproduce this column exactly.

A variant whose API call fails gets NaN and the error text; quota errors (HTTP 429,
gRPC status 8) are retried after a pause. Results are appended as they arrive, so an
interrupted run resumes with --resume.

Usage:
  python alphagenome_splicing.py --input-csv variants.csv --output-csv out.csv --resume
The input needs chr, pos, ref, alt (hg38).
"""
import argparse
import time
from pathlib import Path

import pandas as pd
from alphagenome.data import genome
from alphagenome.models import dna_client, variant_scorers

OUTPUTS = ("SPLICE_SITES", "SPLICE_SITE_USAGE", "SPLICE_JUNCTIONS")
WEIGHTS = {"SPLICE_SITES": 1.0, "SPLICE_SITE_USAGE": 1.0, "SPLICE_JUNCTIONS": 0.2}


def splicing_score(model, scorers, chrom, pos, ref, alt):
    variant = genome.Variant(chromosome=chrom, position=pos,
                             reference_bases=ref, alternate_bases=alt)
    interval = variant.reference_interval.resize(dna_client.SEQUENCE_LENGTH_1MB)
    scores = model.score_variant(interval=interval, variant=variant,
                                 variant_scorers=scorers,
                                 organism=dna_client.Organism.HOMO_SAPIENS)
    tidy = variant_scorers.tidy_scores([scores])
    if tidy is None or tidy.empty:
        return 0.0
    best = tidy.groupby("output_type")["raw_score"].max()
    return float(sum(w * best.get(o, 0.0) for o, w in WEIGHTS.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--api-key-file", default=str(Path.home() / ".config/alphagenome/api_key"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--quota-pause", type=float, default=30.0)
    a = ap.parse_args()

    out = Path(a.output_csv)
    if out.exists() and not a.resume:
        out.unlink()
    done = set()
    if out.exists() and out.stat().st_size:
        done = set(pd.read_csv(out, usecols=["input_row"])["input_row"].astype(int))

    model = dna_client.create(Path(a.api_key_file).read_text().strip())
    scorers = [variant_scorers.RECOMMENDED_VARIANT_SCORERS[o] for o in OUTPUTS]
    data = pd.read_csv(a.input_csv)
    for i, row in data.iterrows():
        if i in done:
            continue
        chrom = str(row["chr"]) if str(row["chr"]).startswith("chr") else f"chr{row['chr']}"
        record = {"input_row": i, "chr": chrom, "pos": int(row["pos"]),
                  "ref": str(row["ref"]).upper(), "alt": str(row["alt"]).upper(),
                  "alphagenome_splicing": float("nan"), "error": ""}
        while True:
            try:
                record["alphagenome_splicing"] = splicing_score(
                    model, scorers, chrom, record["pos"], record["ref"], record["alt"])
                break
            except Exception as exc:
                msg = str(exc)
                if "Quota exceeded" in msg or "429" in msg or "status:8" in msg:
                    time.sleep(a.quota_pause)
                    continue
                record["error"] = f"{type(exc).__name__}: {msg}"[:500]
                break
        pd.DataFrame([record]).to_csv(out, mode="a", index=False,
                                      header=not out.exists() or out.stat().st_size == 0)


if __name__ == "__main__":
    main()
