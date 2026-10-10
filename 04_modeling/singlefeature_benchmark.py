"""
Single-feature benchmark over the full candidate pool, with no model fitted on
AlphaGenome output.

Why this file exists
--------------------
`pipeline.py` scores every candidate feature by fitting a one-feature logistic
regression inside each training fold and evaluating it on the held-out fold.
For the `alphagenome_*` columns that is a machine-learning model trained on
AlphaGenome output, which the AlphaGenome terms ask users not to build.

It is also redundant. A logistic regression on one feature is a monotone map, so
AUROC and AUPRC -- both rank statistics -- are identical whether the raw column
or its logistic transform is scored. This module reproduces the published
per-fold numbers exactly while replacing the fit, for AlphaGenome columns only,
with:

    median-impute (training fold) -> mid-rank ECDF percentile against the
    training fold -> flip if the column is oriented the wrong way round

The flip is a single documented bit per column, stated in ORIENTATION below and
checked against the data rather than fitted per fold. Everything else about the
benchmark is unchanged, and non-AlphaGenome features keep the original recipe so
their published numbers are untouched.

The MERGE ensemble never sees any of this: it is trained on the pool with every
`alphagenome_*` column removed. AlphaGenome appears here only as a baseline the
ensemble is measured against, in the same way CADD and SpliceAI do in
`07_baselines/`.

Usage: python singlefeature_benchmark.py <outdir> [task ...]
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RANDOM_STATE, DATA, FOLDS, TASKS, ID_COLS, MIN_COVERAGE

RS = RANDOM_STATE

# Direction of every AlphaGenome summary column, stated rather than fitted.
# AlphaGenome reports a per-track effect size and its quantile against a
# background distribution; the max/mean summaries grow with disruption, the min
# summaries shrink with it, and the splice channel grows with it.
ORIENTATION = {
    "alphagenome_splicing":              +1,
    "alphagenome_raw_score_max":         +1,
    "alphagenome_raw_score_mean":        +1,
    "alphagenome_raw_score_min":         -1,
    "alphagenome_quantile_score_max":    +1,
    "alphagenome_quantile_score_mean":   +1,
    "alphagenome_quantile_score_min":    -1,
}


def is_alphagenome(col):
    return col.startswith("alphagenome")


def load(task):
    df = pd.read_csv(os.path.join(DATA, TASKS[task]), low_memory=False, dtype={"chr": str})
    df["chr"] = df["chr"].str.replace("chr", "", regex=False)
    feats = [c for c in df.columns if c not in ID_COLS]
    keep = [c for c in feats
            if df[c].notna().mean() >= MIN_COVERAGE and df[c].nunique(dropna=True) > 1]
    df["fold"] = -1
    for k, chroms in FOLDS.items():
        df.loc[df["chr"].isin(chroms), "fold"] = k
    return df[df["fold"] != -1].reset_index(drop=True), keep


def midrank_percentile(values, reference):
    ref = np.sort(np.asarray(reference, dtype=float))
    v = np.asarray(values, dtype=float)
    lo = np.searchsorted(ref, v, side="left")
    hi = np.searchsorted(ref, v, side="right")
    return (lo + hi) / (2.0 * len(ref))


def score_fold(df, feat, tr, te, y):
    """Held-out-fold scores for one feature, fitted on the training fold only."""
    if is_alphagenome(feat):
        if feat not in ORIENTATION:
            raise SystemExit(f"no stated orientation for {feat}")
        med = float(np.nanmedian(df.loc[tr, feat].values))
        xtr = np.nan_to_num(df.loc[tr, feat].values.astype(float), nan=med)
        xte = np.nan_to_num(df.loc[te, feat].values.astype(float), nan=med)
        p = midrank_percentile(xte, xtr)
        return p if ORIENTATION[feat] > 0 else 1.0 - p
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()),
                     ("clf", LogisticRegression(class_weight="balanced", max_iter=3000,
                                                random_state=RS))])
    pipe.fit(df.loc[tr, [feat]].values, y[tr])
    return pipe.predict_proba(df.loc[te, [feat]].values)[:, 1]


def run(outdir, task):
    os.makedirs(outdir, exist_ok=True)
    df, feats = load(task)
    y = df["label"].values.astype(int)
    folds = sorted(FOLDS)
    rows, oof_tbl = [], df[ID_COLS].copy()
    for f in feats:
        oof = np.full(len(df), np.nan)
        per = []
        for k in folds:
            te = df["fold"].values == k
            tr = ~te
            if len(np.unique(y[tr])) < 2:
                continue
            p = score_fold(df, f, tr, te, y)
            oof[te] = p
            per.append((roc_auc_score(y[te], p), average_precision_score(y[te], p)))
        a = np.array(per)
        # sample SD over the three outer folds (divisor k - 1), as for MERGE
        rows.append({"feature": f, "auroc_mean": a[:, 0].mean(), "auroc_sd": a[:, 0].std(ddof=1),
                     "auprc_mean": a[:, 1].mean(), "auprc_sd": a[:, 1].std(ddof=1),
                     "scored_by": "rank" if is_alphagenome(f) else "logistic"})
        oof_tbl[f] = oof
    sf = pd.DataFrame(rows).sort_values("auroc_mean", ascending=False)
    sf.to_csv(f"{outdir}/{task}_singlefeature.csv", index=False)
    oof_tbl.to_csv(f"{outdir}/{task}_singlefeature_oof.csv", index=False)
    top = sf.iloc[0]
    nonag = sf[~sf["feature"].map(is_alphagenome)].iloc[0]
    print(f"[{task}] n={len(df)} feats={len(feats)}  best={top.feature} "
          f"{top.auroc_mean:.4f}/{top.auprc_mean:.4f}  "
          f"best non-AlphaGenome={nonag.feature} {nonag.auroc_mean:.4f}/{nonag.auprc_mean:.4f}",
          flush=True)


if __name__ == "__main__":
    outdir = sys.argv[1]
    for t in (sys.argv[2:] or list(TASKS)):
        run(outdir, t)
