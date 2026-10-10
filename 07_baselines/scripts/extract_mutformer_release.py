"""Extract MutFormer hg38 scores from the official release without unpacking it.

The official archive contains one 2.8 GB tab-delimited file with columns:
chromosome, start, end, reference allele, alternate allele, MutFormer score.
Only variants present in the MERGE coding benchmark are retained.
"""
from __future__ import annotations

import csv
import os
import sys
import gzip
import zipfile

import pandas as pd


def main() -> None:
    archive, variants_csv, output_tsv = sys.argv[1:4]
    variants = pd.read_csv(
        variants_csv,
        usecols=["chr", "pos", "ref", "alt"],
        low_memory=False,
    )
    variants["chr"] = (
        variants["chr"].astype(str).str.replace("chr", "", regex=False)
    )
    targets = {
        (str(chrom), int(pos), str(ref), str(alt))
        for chrom, pos, ref, alt in variants.itertuples(index=False, name=None)
    }

    scores: dict[tuple[str, int, str, str], float] = {}
    lines = 0
    if archive.lower().endswith(".gz"):
        source = gzip.open(archive, "rb")
    else:
        zf = zipfile.ZipFile(archive)
        members = zf.namelist()
        if members != ["hg38_MutFormer.txt"]:
            zf.close()
            raise RuntimeError(f"Unexpected archive contents: {members}")
        source = zf.open(members[0])
    try:
        with source as handle:
            for raw in handle:
                lines += 1
                fields = raw.rstrip(b"\r\n").split(b"\t")
                if len(fields) != 6:
                    continue
                try:
                    chrom = fields[0].decode().removeprefix("chr")
                    pos = int(fields[1])
                    ref = fields[3].decode()
                    alt = fields[4].decode()
                    score = float(fields[5])
                except (UnicodeDecodeError, ValueError):
                    continue
                key = (chrom, pos, ref, alt)
                if key in targets:
                    # Defensive maximum if the source ever contains duplicate keys.
                    scores[key] = max(score, scores.get(key, float("-inf")))
                if lines % 10_000_000 == 0:
                    print(f"scanned={lines:,} matched={len(scores):,}", flush=True)
    finally:
        if "zf" in locals():
            zf.close()

    os.makedirs(os.path.dirname(output_tsv), exist_ok=True)
    with open(output_tsv, "w", newline="") as out:
        writer = csv.writer(out, delimiter="\t", lineterminator="\n")
        writer.writerow(["chr", "pos", "ref", "alt", "score"])
        for key in sorted(scores, key=lambda k: (k[0], k[1], k[2], k[3])):
            writer.writerow([*key, scores[key]])

    total = len(targets)
    print(
        f"MutFormer: {len(scores):,}/{total:,} unique variants "
        f"({len(scores) / total:.1%}) -> {output_tsv}"
    )


if __name__ == "__main__":
    main()
