"""Pull the coding baselines out of dbNSFP.

Runs where the dbNSFP bgzip file and its index live. Columns are located by
name in the header, not by hard-coded position, so a dbNSFP version bump does
not silently shift the values.

Several dbNSFP scores are oriented "lower is more damaging" (SIFT, SIFT4G,
PROVEAN, FATHMM). Those are negated here so that every emitted score means
"larger is more likely pathogenic", which is what evaluate.py assumes. Fields
that carry one value per transcript, separated by ';', are reduced to the most
damaging value after that orientation is applied.

Usage: python extract_dbnsfp.py <dbnsfp.gz> <variants.csv> <out-dir> [task]
"""
import os
import sys

import pandas as pd
import pysam

# emitted name -> dbNSFP column name
WANTED = {
    "SIFT4G": "SIFT4G_score",
    "Polyphen2_HVAR": "Polyphen2_HVAR_score",
    "MutationAssessor": "MutationAssessor_score",
    "PROVEAN": "PROVEAN_score",
    "PrimateAI": "PrimateAI_score",
    "GERP++": "GERP++_RS",
    "phyloP100way_dbnsfp": "phyloP100way_vertebrate",
    "phyloP470way_dbnsfp": "phyloP470way_mammalian",
    "VEST4": "VEST4_score",
    "MetaRNN": "MetaRNN_score",
    "M-CAP": "M-CAP_score",
    "REVEL": "REVEL_score",
    "BayesDel": "BayesDel_noAF_score",
    "ClinPred": "ClinPred_score",
    "VARITY": "VARITY_R_score",
    "CADD_dbnsfp": "CADD_phred",
}
# dbNSFP scores where a smaller number means more damaging.
FLIP = {"SIFT_score", "SIFT4G_score", "PROVEAN_score", "FATHMM_score"}


def parse(raw, flip):
    """Most damaging numeric value in a possibly ';'-separated dbNSFP field."""
    best = None
    for part in str(raw).split(";"):
        part = part.strip()
        if part in ("", ".", "-"):
            continue
        try:
            v = float(part)
        except ValueError:
            continue
        if flip:
            v = -v
        best = v if best is None else max(best, v)
    return best


def main():
    dbnsfp, variants, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    task = sys.argv[4] if len(sys.argv) > 4 else "coding"
    os.makedirs(outdir, exist_ok=True)

    tbx = pysam.TabixFile(dbnsfp)
    header = tbx.header[0].lstrip("#").split("\t")
    header[0] = "chr"
    idx = {}
    for out_name, col in WANTED.items():
        if col not in header:
            print(f"  !! {col} not in this dbNSFP header, skipping {out_name}")
            continue
        idx[out_name] = (header.index(col), col in FLIP)
    i_ref, i_alt = header.index("ref"), header.index("alt")
    prefix = "chr" if any(c.startswith("chr") for c in tbx.contigs) else ""

    df = pd.read_csv(variants, usecols=["chr", "pos", "ref", "alt"], low_memory=False)
    df["chr"] = df["chr"].astype(str).str.replace("chr", "", regex=False)
    df = df.drop_duplicates().sort_values(["chr", "pos"]).reset_index(drop=True)

    got = {name: {} for name in idx}
    known = set(tbx.contigs)
    for n, (chrom, pos, ref, alt) in enumerate(
            zip(df["chr"], df["pos"], df["ref"], df["alt"])):
        contig = prefix + chrom
        if contig not in known:
            continue
        try:
            rows = list(tbx.fetch(contig, int(pos) - 1, int(pos)))
        except (ValueError, OSError):
            continue
        for row in rows:
            f = row.split("\t")
            if f[i_ref] != ref or f[i_alt] != alt:
                continue
            for name, (i, flip) in idx.items():
                v = parse(f[i], flip) if i < len(f) else None
                if v is not None:
                    got[name][(chrom, pos, ref, alt)] = v
            break
        if (n + 1) % 2000 == 0:
            print(f"  {n + 1}/{len(df)}", flush=True)

    keys = list(zip(df["chr"], df["pos"], df["ref"], df["alt"]))
    for name, lookup in got.items():
        out = df.copy()
        out["score"] = [lookup.get(k) for k in keys]
        out = out.dropna(subset=["score"])
        p = os.path.join(outdir, f"{task}__{name}.tsv")
        out.to_csv(p, sep="\t", index=False)
        print(f"{name}: {len(out)}/{len(df)} ({len(out) / len(df):.1%}) -> {p}")


if __name__ == "__main__":
    main()
