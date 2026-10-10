"""Look up a bigWig track at each variant position.

Kept separate from extract_noncoding.py so it can be run wherever the bigWig
happens to live. The phyloP file is 9.87 GB and the host that can fetch it
quickly is not the host that runs the modelling; moving 9.87 GB between them
over a double ssh hop runs at ~135 KB/s, while moving the 2 MB of results back
is instant.

Usage: python extract_bigwig.py <track.bw> <tool-name> <sites.csv> <out-dir> [task]
"""
import os
import sys

import pandas as pd
import pyBigWig


def main():
    track, tool, sites_csv, outdir = sys.argv[1:5]
    task = sys.argv[5] if len(sys.argv) > 5 else "noncoding"
    os.makedirs(outdir, exist_ok=True)

    df = pd.read_csv(sites_csv, usecols=["chr", "pos", "ref", "alt"], low_memory=False)
    df["chr"] = df["chr"].astype(str).str.replace("chr", "", regex=False)
    df = df.drop_duplicates().sort_values(["chr", "pos"]).reset_index(drop=True)

    bw = pyBigWig.open(track)
    chroms = bw.chroms()
    prefix = "chr" if any(c.startswith("chr") for c in chroms) else ""
    got = {}
    for n, (chrom, pos) in enumerate(zip(df["chr"], df["pos"])):
        contig = prefix + chrom
        pos = int(pos)
        if contig not in chroms or pos > chroms[contig]:
            continue
        try:
            v = bw.values(contig, pos - 1, pos)[0]
        except (RuntimeError, IndexError):
            continue
        if v is not None and v == v:          # not NaN
            got[(chrom, pos)] = float(v)
        if (n + 1) % 20000 == 0:
            print(f"  {n + 1}/{len(df)}", flush=True)
    bw.close()

    df["score"] = [got.get((c, int(p))) for c, p in zip(df["chr"], df["pos"])]
    out = df.dropna(subset=["score"])
    path = os.path.join(outdir, f"{task}__{tool}.tsv")
    out.to_csv(path, sep="\t", index=False)
    print(f"{tool}: {len(out)}/{len(df)} scored "
          f"({len(out) / max(len(df), 1):.1%}) -> {path}")


if __name__ == "__main__":
    main()
