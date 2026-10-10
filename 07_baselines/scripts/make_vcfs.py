"""Write the benchmark datasets as VCFs for the external baseline services.

Each output file is kept under the CADD web service's 2 MB / 100,000-variant
upload limit, so every dataset is one or two submissions. Coordinates are
GRCh38 with Ensembl-style chromosome names (no "chr" prefix), sorted by
chromosome then position, one row per unique (chr, pos, ref, alt) site.
The VCFs are inputs for CADD, SpliceAI and Pangolin and are not committed.

Usage: python make_vcfs.py [repo-root] [out-dir]
"""
import os
import sys

import pandas as pd

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "vcf")

TASKS = {"coding": "coding_features.csv", "splicing": "splicing_features.csv",
         "noncoding": "noncoding_features.csv", "nonsnv": "non_snv_features.csv"}
CHROM_ORDER = {c: i for i, c in enumerate(
    [str(i) for i in range(1, 23)] + ["X", "Y", "MT", "M"])}
MAX_BYTES = 1_700_000          # headroom under the 2 MB limit
HEADER = "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"


def main():
    os.makedirs(OUT, exist_ok=True)
    for task, fname in TASKS.items():
        df = pd.read_csv(os.path.join(ROOT, "03_datasets", fname),
                         usecols=["chr", "pos", "ref", "alt"], low_memory=False)
        df["chr"] = df["chr"].astype(str).str.replace("chr", "", regex=False)
        df = df.drop_duplicates()
        df["_k"] = df["chr"].map(CHROM_ORDER).fillna(99).astype(int)
        df = df.sort_values(["_k", "pos"])
        rows = [f"{c}\t{p}\t.\t{r}\t{a}\t.\t.\t.\n"
                for c, p, r, a in zip(df["chr"], df["pos"], df["ref"], df["alt"])]
        nparts = max(1, -(-sum(len(r) for r in rows) // MAX_BYTES))
        step = -(-len(rows) // nparts)
        for i in range(nparts):
            chunk = rows[i * step:(i + 1) * step]
            if not chunk:
                continue
            name = f"{task}.vcf" if nparts == 1 else f"{task}.part{i + 1}.vcf"
            path = os.path.join(OUT, name)
            with open(path, "w") as fh:
                fh.write(HEADER)
                fh.writelines(chunk)
            print(f"{name}: {len(chunk)} variants, "
                  f"{os.path.getsize(path) / 1e6:.2f} MB")
        print(f"  [{task}] {len(df)} unique sites")


if __name__ == "__main__":
    main()
