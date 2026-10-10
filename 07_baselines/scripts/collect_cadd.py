"""Turn the CADD service output into 07_baselines/scores/<task>__CADD.tsv.

CADD returns "#Chrom Pos Ref Alt RawScore PHRED"; PHRED is the scaled rank and
is the value used for the comparison. Datasets split across several upload
parts are concatenated back together here.

Usage: python collect_cadd.py <cadd-dir> <out-dir>
"""
import glob
import gzip
import os
import re
import sys

import pandas as pd


def read_one(path):
    rows = []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 6:
                continue
            rows.append((f[0], int(f[1]), f[2], f[3], float(f[5])))
    return pd.DataFrame(rows, columns=["chr", "pos", "ref", "alt", "score"])


def main():
    cadd_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    parts = {}
    for p in sorted(glob.glob(os.path.join(cadd_dir, "*.cadd.tsv.gz"))):
        name = os.path.basename(p)[:-len(".cadd.tsv.gz")]
        task = re.sub(r"\.part\d+$", "", name)
        parts.setdefault(task, []).append(p)
    for task, files in sorted(parts.items()):
        df = pd.concat([read_one(p) for p in files], ignore_index=True)
        df = df.drop_duplicates(subset=["chr", "pos", "ref", "alt"])
        out = os.path.join(out_dir, f"{task}__CADD.tsv")
        df.to_csv(out, sep="\t", index=False)
        print(f"{task}: {len(df)} scored from {len(files)} part(s) -> {out}")


if __name__ == "__main__":
    main()
