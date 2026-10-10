"""Look up the non-coding baseline scores for a dataset's variants.

Writes one 07_baselines/scores/<task>__<tool>.tsv per tool in the format
evaluate.py expects (chr, pos, ref, alt, score).

The tabix files are read with `tabix -R`, which walks the sorted region list in
a single pass. Fetching each site individually is the obvious alternative and is
far slower here: these files sit on NFS, where tens of thousands of independent
random reads cost more than one ordered sweep.

The sources disagree about layout, so each declares its own:

  bed   chrom, start, end, score   0-based half-open; a variant at 1-based POS
                                   falls in the interval with start <= POS-1 < end
  tsv   chrom, pos, score          1-based, one row per base

Chromosome naming is taken from the file rather than assumed.

Usage: python extract_noncoding.py <task> <variants.csv> <data-dir> <out-dir>
"""
import os
import subprocess
import sys
import tempfile

import pandas as pd

SOURCES = {
    # tool          file                        layout  score column (0-based)
    "LINSIGHT":  ("GRCh38_LinSight.bed.gz",  "bed", 3),
    # v0.3.1 is the copy bundled with GREEN-VARAN; v0.4 is the release retrained
    # on GRCh38. Both are reported, because comparing only against the older one
    # would understate the strongest-motivated non-coding comparator.
    "ReMM_v031": ("GRCh38_ReMM.tsv.gz",      "tsv", 2),
    "ReMM_v04":  ("ReMM.v0.4.hg38.tsv.gz",   "tsv", 2),
    "ncER":      ("GRCh38_ncER_perc.bed.gz", "bed", 3),
}
BIGWIGS = {"phyloP100way": "hg38.phyloP100way.bw"}
MAX_INTERVAL = 100_000      # guard against a pathological interval


def file_uses_chr_prefix(path):
    out = subprocess.run(["tabix", "-l", path], capture_output=True, text=True)
    return any(c.startswith("chr") for c in out.stdout.split())


def extract_tabix(path, layout, score_col, sites):
    """sites: {chrom_without_prefix: sorted list of 1-based positions}"""
    prefix = "chr" if file_uses_chr_prefix(path) else ""
    wanted = {(c, p) for c, ps in sites.items() for p in ps}
    with tempfile.NamedTemporaryFile("w", suffix=".bed", delete=False) as fh:
        region_file = fh.name
        for chrom in sorted(sites):
            for pos in sites[chrom]:
                fh.write(f"{prefix}{chrom}\t{pos - 1}\t{pos}\n")

    out = {}
    proc = subprocess.Popen(["tabix", "-R", region_file, path],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            text=True, bufsize=1 << 20)
    for line in proc.stdout:
        f = line.rstrip("\n").split("\t")
        if len(f) <= score_col:
            continue
        chrom = f[0][3:] if f[0].startswith("chr") else f[0]
        try:
            value = float(f[score_col])
        except ValueError:
            continue
        if layout == "tsv":
            positions = (int(f[1]),)
        else:
            start, end = int(f[1]), int(f[2])
            if end - start > MAX_INTERVAL:
                continue
            positions = range(start + 1, end + 1)   # 0-based half-open -> 1-based
        for pos in positions:
            key = (chrom, pos)
            if key in wanted:
                prev = out.get(key)
                out[key] = value if prev is None else max(prev, value)
    proc.wait()
    os.unlink(region_file)
    return out


def extract_bigwig(path, sites):
    import pyBigWig
    bw = pyBigWig.open(path)
    chroms = bw.chroms()
    prefix = "chr" if any(c.startswith("chr") for c in chroms) else ""
    out = {}
    for chrom, positions in sites.items():
        contig = prefix + chrom
        if contig not in chroms:
            continue
        limit = chroms[contig]
        for pos in positions:
            if pos > limit:
                continue
            try:
                v = bw.values(contig, pos - 1, pos)[0]
            except (RuntimeError, IndexError):
                continue
            if v is not None and v == v:        # not NaN
                out[(chrom, pos)] = float(v)
    bw.close()
    return out


def write(outdir, task, tool, df, got):
    out = df.copy()
    out["score"] = [got.get((c, int(p))) for c, p in zip(df["chr"], df["pos"])]
    out = out.dropna(subset=["score"])
    path = os.path.join(outdir, f"{task}__{tool}.tsv")
    out.to_csv(path, sep="\t", index=False)
    print(f"  {tool}: {len(out)}/{len(df)} scored "
          f"({len(out) / max(len(df), 1):.1%}) -> {path}", flush=True)


def main():
    task, variants, datadir, outdir = sys.argv[1:5]
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(variants, usecols=["chr", "pos", "ref", "alt"], low_memory=False)
    df["chr"] = df["chr"].astype(str).str.replace("chr", "", regex=False)
    df = df.drop_duplicates().sort_values(["chr", "pos"]).reset_index(drop=True)

    # These are single-base genome scores, so the key ignores the alleles.
    sites = {}
    for chrom, pos in zip(df["chr"], df["pos"]):
        sites.setdefault(chrom, set()).add(int(pos))
    sites = {c: sorted(p) for c, p in sites.items()}
    n_sites = sum(len(p) for p in sites.values())
    print(f"[{task}] {len(df)} variants at {n_sites} sites", flush=True)

    for tool, (fname, layout, col) in SOURCES.items():
        path = os.path.join(datadir, fname)
        if not os.path.exists(path):
            print(f"  {tool}: {path} missing, skipped", flush=True)
            continue
        write(outdir, task, tool, df, extract_tabix(path, layout, col, sites))

    for tool, fname in BIGWIGS.items():
        path = os.path.join(datadir, fname)
        if not os.path.exists(path) or os.path.exists(path + ".aria2"):
            print(f"  {tool}: {fname} not ready, skipped", flush=True)
            continue
        write(outdir, task, tool, df, extract_bigwig(path, sites))


if __name__ == "__main__":
    main()
