"""Attach official GPN-MSA scores to a variant table by exact (chr, pos, ref, alt).

GPN-MSA is not run here: its genome-wide hg38 scores are published as a
bgzipped, tabix-indexed table (Hugging Face dataset songlab/gpn-msa-hg38-scores,
file scores.tsv.bgz; columns chrom, pos, ref, alt, score; chromosomes without a
"chr" prefix). Each variant is looked up at its position and matched on REF and
ALT; a variant without an entry gets NaN.

Use a local copy of the table for large inputs. The table can also be read over
HTTPS by htslib range requests (set CURL_CA_BUNDLE if htslib does not find your
CA certificates); results are written as they are found, so an interrupted run
resumes where it stopped.

Usage:
  python match_gpn_msa.py --input variants.csv --output variants_gpn.csv \
      [--table scores.tsv.bgz] [--index scores.tsv.bgz.tbi]

The input needs columns chr, pos, ref, alt (chr with or without "chr"); the
output is the input with a GPN_MSA_score column added or replaced.
"""
import argparse
import os

import pandas as pd
import pysam

TABLE = ("https://huggingface.co/datasets/songlab/gpn-msa-hg38-scores/resolve/main/"
         "scores.tsv.bgz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--table", default=TABLE)
    ap.add_argument("--index", default=None, help="local .tbi (default: <table>.tbi)")
    a = ap.parse_args()

    df = pd.read_csv(a.input, low_memory=False)
    progress = a.output + ".partial"
    found = {}
    if os.path.exists(progress):
        for line in open(progress):
            i, _, v = line.rstrip("\n").partition(",")
            found[int(i)] = float(v) if v else None

    tb = pysam.TabixFile(a.table, index=a.index)
    chrom = df["chr"].astype(str).str.replace("chr", "", regex=False)
    # Visit variants in genomic order: neighbouring lookups then share the
    # compressed blocks htslib has already fetched, which matters a lot remotely.
    order = sorted(df.index, key=lambda i: (chrom[i], int(df.at[i, "pos"])))
    with open(progress, "a") as fh:
        for i in order:
            c, pos, ref, alt = chrom[i], int(df.at[i, "pos"]), df.at[i, "ref"], df.at[i, "alt"]
            if i in found:
                continue
            score = None
            for line in tb.fetch(c, pos - 1, pos):
                f = line.split("\t")
                if int(f[1]) == pos and f[2] == ref and f[3] == alt:
                    score = float(f[4])
            found[i] = score
            fh.write(f"{i},{'' if score is None else score}\n")
            fh.flush()

    df["GPN_MSA_score"] = pd.Series(found).reindex(df.index).astype(float)
    df.to_csv(a.output, index=False)
    os.remove(progress)
    print(f"{len(df)} variants, {df['GPN_MSA_score'].notna().sum()} matched -> {a.output}")


if __name__ == "__main__":
    main()
