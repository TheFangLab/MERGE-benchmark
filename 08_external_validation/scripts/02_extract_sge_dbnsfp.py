"""
Extract the established-predictor scores for the SGE cohort from dbNSFP 5.3a.

Variants are matched on hg38 chromosome, position, REF and ALT in the chr17 file of
the academic dbNSFP 5.3a release (not redistributed; see 00_raw_data/). SIFT4G,
PROVEAN and ESM-1b are negated so that larger means more damaging.

Usage: python 02_extract_sge_dbnsfp.py --dbnsfp dbNSFP5.3a.zip
"""
from __future__ import annotations

import argparse
import gzip
import zipfile
from pathlib import Path

import pandas as pd

EXP = Path(__file__).resolve().parent.parent
COHORT = EXP / "cohort" / "sge_merge_external_validation_hg38.csv"
MEMBER = "dbNSFP5.3a/dbNSFP5.3a_variant.chr17.gz"
OUT = EXP / "model_outputs" / "sge_dbnsfp_scores.csv"
ARCHIVE = None                                  # set from the command line in main()

WANTED = {
    "SIFT4G": "SIFT4G_score",
    "PolyPhen-2 HVAR": "Polyphen2_HVAR_score",
    "MutationAssessor": "MutationAssessor_score",
    "PROVEAN": "PROVEAN_score",
    "PrimateAI": "PrimateAI_score",
    "VEST4": "VEST4_score",
    "MetaRNN": "MetaRNN_score",
    "M-CAP": "M-CAP_score",
    "REVEL": "REVEL_score",
    "BayesDel": "BayesDel_noAF_score",
    "ClinPred": "ClinPred_score",
    "VARITY": "VARITY_R_score",
    "CADD": "CADD_phred",
    "GERP++": "GERP++_RS",
    "phyloP 100-way": "phyloP100way_vertebrate",
    "phyloP 470-way": "phyloP470way_mammalian",
    "AlphaMissense": "AlphaMissense_score",
    "ESM-1b": "ESM1b_score",
    "MutFormer": "MutFormer_score",
}
FLIP = {"SIFT4G_score", "PROVEAN_score", "ESM1b_score"}
AUDIT = ["clinvar_id", "clinvar_clnsig", "clinvar_review", "genename", "aapos"]


def most_damaging(raw: str, flip: bool):
    values = []
    for part in str(raw).split(";"):
        part = part.strip()
        if part in {"", ".", "-"}:
            continue
        try:
            value = float(part)
        except ValueError:
            continue
        values.append(-value if flip else value)
    return max(values) if values else None


def main():
    global ARCHIVE
    ap = argparse.ArgumentParser()
    ap.add_argument("--dbnsfp", required=True, type=Path, help="dbNSFP5.3a.zip")
    ARCHIVE = ap.parse_args().dbnsfp
    cohort = pd.read_csv(COHORT, low_memory=False)
    cohort["chr"] = cohort["chr"].astype(str).str.removeprefix("chr")
    targets = {
        (str(c), int(p), str(r), str(a)): i
        for i, (c, p, r, a) in enumerate(
            cohort[["chr", "pos", "ref", "alt"]].itertuples(index=False, name=None)
        )
    }
    min_pos = min(k[1] for k in targets)
    max_pos = max(k[1] for k in targets)
    found = {}
    scanned = 0

    with zipfile.ZipFile(ARCHIVE) as archive:
        with archive.open(MEMBER) as compressed:
            with gzip.open(compressed, "rt", encoding="utf-8", errors="replace") as handle:
                header = handle.readline().rstrip("\n").lstrip("#").split("\t")
                idx = {name: header.index(col) for name, col in WANTED.items()}
                audit_idx = {col: header.index(col) for col in AUDIT}
                for line in handle:
                    scanned += 1
                    f = line.rstrip("\n").split("\t")
                    try:
                        pos = int(f[1])
                    except (ValueError, IndexError):
                        continue
                    if pos < min_pos:
                        continue
                    if pos > max_pos:
                        break
                    key = (f[0].removeprefix("chr"), pos, f[2], f[3])
                    if key not in targets:
                        continue
                    row = {"chr": key[0], "pos": pos, "ref": key[2], "alt": key[3]}
                    for name, col in WANTED.items():
                        row[name] = most_damaging(f[idx[name]], col in FLIP)
                    for col in AUDIT:
                        row[col] = f[audit_idx[col]]
                    found[key] = row
                    if len(found) % 200 == 0:
                        print(f"matched={len(found)}/{len(targets)} scanned={scanned:,}", flush=True)

    scores = pd.DataFrame(found.values())
    merged = cohort[["variant_id", "label", "chr", "pos", "ref", "alt"]].merge(
        scores, on=["chr", "pos", "ref", "alt"], how="left", validate="one_to_one"
    )
    merged.to_csv(OUT, index=False)
    print(f"matched rows: {len(found)}/{len(targets)}; scanned={scanned:,}")
    for name in WANTED:
        print(f"{name}: {merged[name].notna().sum()}/{len(merged)}")


if __name__ == "__main__":
    main()
