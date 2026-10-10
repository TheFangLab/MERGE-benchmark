"""Write AlphaGenome's out-of-fold benchmark score as comparator score files.

AlphaGenome is an independent comparator here: MERGE is trained on the candidate
pool with every `alphagenome_*` column removed, so AlphaGenome is not one of the
ensemble's inputs.

The scores are read from 05_results/<task>_singlefeature_oof.csv, the
chromosome-held-out output of 04_modeling/singlefeature_benchmark.py: missing
scores take the training-fold median, and each held-out fold is mapped to its
mid-rank percentile against the training folds, oriented by a stated direction.
Nothing is fitted to labels, and the training-fold imputation is retained here,
so these files cover every benchmark variant.

Columns used (manuscript Methods, "Comparison with established predictors"):

    coding      AlphaGenome               the AUROC-leading AlphaGenome summary
    nonsnv      AlphaGenome               the AUROC-leading AlphaGenome summary
    splicing    AlphaGenome               alphagenome_splicing
    noncoding   AlphaGenome_raw_max       alphagenome_raw_score_max
                AlphaGenome_quantile_max  alphagenome_quantile_score_max

The two non-coding maxima are reported separately because they rank variants
almost identically (AUROC) but differ in AUPRC.

Usage: python extract_alphagenome_feature.py [repo-root]
"""
import os
import sys

import pandas as pd

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(ROOT, "07_baselines", "scores")

FIXED = {
    "splicing": {"AlphaGenome": "alphagenome_splicing"},
    "noncoding": {"AlphaGenome_raw_max": "alphagenome_raw_score_max",
                  "AlphaGenome_quantile_max": "alphagenome_quantile_score_max"},
}


def columns(task):
    if task in FIXED:
        return FIXED[task]
    sf = pd.read_csv(os.path.join(ROOT, "05_results", f"{task}_singlefeature.csv"))
    ag = sf[sf.feature.str.startswith("alphagenome")]
    return {"AlphaGenome": ag.sort_values("auroc_mean", ascending=False).iloc[0].feature}


def main():
    os.makedirs(OUT, exist_ok=True)
    for task in ["coding", "splicing", "noncoding", "nonsnv"]:
        oof = pd.read_csv(os.path.join(ROOT, "05_results", f"{task}_singlefeature_oof.csv"),
                          low_memory=False)
        oof["chr"] = oof["chr"].astype(str).str.replace("chr", "", regex=False)
        for tool, col in columns(task).items():
            out = os.path.join(OUT, f"{task}__{tool}.tsv")
            d = oof[["chr", "pos", "ref", "alt", col]].rename(columns={col: "score"})
            d.dropna(subset=["score"]).to_csv(out, sep="\t", index=False)
            print(f"[{task}] {tool} = {col} -> {out} ({d['score'].notna().sum()} scored)")


if __name__ == "__main__":
    main()
