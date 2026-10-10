"""
Gene-level exclusion: retrain MERGE-Coding without any BRCA1 variant and re-test it
on the SGE cohort.

Every coding training variant in the BRCA1 gene body (hg38 chr17:43,044,295-
43,125,364; 212 variants) is removed, leaving 4,077. Feature selection, algorithm
choice and fitting are then repeated from scratch with the benchmark procedure
(04_modeling/export_models.py, beam search as selected for coding), on the candidate
pool without AlphaGenome. The resulting model is applied unchanged to the SGE cohort
and compared with the frozen MERGE-Coding model on the 1,421-variant common subset
using 2,000 paired, class-stratified bootstrap resamples (numpy default_rng(42)).

Outputs:
  models/MERGE_coding_noBRCA1.pkl
  results/sge_noBRCA1_predictions.csv, sge_noBRCA1_metrics.csv, sge_noBRCA1_meta.json
  results/paired_with_vs_without_BRCA1.csv

Usage: python 05_retrain_without_brca1.py
"""
import json
import os
import sys
import tempfile

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
REPO = os.path.dirname(EXP)
sys.path.insert(0, os.path.join(REPO, "04_modeling"))
from export_models import fit_bundle                              # noqa: E402
from fast_metrics import WeightedScorer                          # noqa: E402

BRCA1 = ("17", 43_044_295, 43_125_364)          # hg38 gene body
NAME = "MERGE without AlphaGenome"
N_BOOT = 2000


def interval(d):
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    train = pd.read_csv(os.path.join(REPO, "03_datasets", "coding_features.csv"),
                        low_memory=False)
    chrom = train["chr"].astype(str).str.replace("chr", "", regex=False)
    in_brca1 = (chrom == BRCA1[0]) & train["pos"].between(BRCA1[1], BRCA1[2])
    kept = train[~in_brca1]
    print(f"coding training table: {len(train)} -> {len(kept)} "
          f"({int(in_brca1.sum())} BRCA1 variants removed)")

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "coding_features_noBRCA1.csv")
        kept.to_csv(path, index=False)
        strat = json.load(open(os.path.join(REPO, "05_results", "coding_meta.json")))["best_strategy"]
        pipe, sel, algo, scores, df, *_ = fit_bundle("coding", strat, path=path)
    bundle = {"task": "coding", "pipeline": pipe, "features": sel, "feature_selection": strat,
              "algorithm": algo, "n_train": int(len(df)),
              "positive_rate": float(df["label"].mean()),
              "algorithm_inner_group_cv_auroc": {k: float(v) for k, v in scores.items()},
              "trained_on": f"03_datasets/coding_features.csv without the "
                            f"{int(in_brca1.sum())} variants in chr17:43,044,295-43,125,364"}
    os.makedirs(os.path.join(EXP, "models"), exist_ok=True)
    joblib.dump(bundle, os.path.join(EXP, "models", "MERGE_coding_noBRCA1.pkl"), compress=3)
    print(f"panel ({len(sel)}): {sel}; algorithm {algo}")

    sge = pd.read_csv(os.path.join(EXP, "features", "sge_features.csv"), low_memory=False)
    y_all = sge["label"].to_numpy(int)
    p_all = pipe.predict_proba(sge[sel].to_numpy())[:, 1]

    # the excluded model on the whole cohort
    rng = np.random.default_rng(42)
    pos, neg = np.flatnonzero(y_all == 1), np.flatnonzero(y_all == 0)
    sc = WeightedScorer(p_all, y_all)
    d = np.array([sc(np.bincount(np.concatenate([rng.choice(pos, pos.size, True),
                                                 rng.choice(neg, neg.size, True)]),
                                 minlength=len(y_all))) for _ in range(N_BOOT)])
    au_lo, au_hi = interval(d[:, 0])
    ap_lo, ap_hi = interval(d[:, 1])
    out = os.path.join(EXP, "results")
    pd.DataFrame([{"predictor": NAME, "n_common": len(y_all),
                   "auroc": roc_auc_score(y_all, p_all), "auroc_ci_low": au_lo,
                   "auroc_ci_high": au_hi, "auprc": average_precision_score(y_all, p_all),
                   "auprc_ci_low": ap_lo, "auprc_ci_high": ap_hi}]
                 ).to_csv(os.path.join(out, "sge_noBRCA1_metrics.csv"), index=False)
    scored = sge[["variant_id", "chr", "pos", "ref", "alt", "label"]].copy()
    scored[NAME] = p_all
    scored.to_csv(os.path.join(out, "sge_noBRCA1_predictions.csv"), index=False)
    json.dump({NAME: {"algorithm": algo, "features": sel, "n_train": int(len(df))},
               "note": bundle["trained_on"]},
              open(os.path.join(out, "sge_noBRCA1_meta.json"), "w"), indent=2)

    # paired with/without BRCA1 on the common subset of the main analysis
    frozen = pd.read_csv(os.path.join(out, "sge_merge_predictions.csv"))
    assert (frozen["variant_id"].values == sge["variant_id"].values).all()
    common = frozen["in_common_subset"].to_numpy(bool)
    y = y_all[common]
    a, b = frozen[NAME].to_numpy(float)[common], p_all[common]
    rng = np.random.default_rng(42)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    sa, sb = WeightedScorer(a, y), WeightedScorer(b, y)
    da, db_ = [], []
    for _ in range(N_BOOT):
        w = np.bincount(np.concatenate([rng.choice(pos, pos.size, True),
                                        rng.choice(neg, neg.size, True)]), minlength=len(y))
        (a1, a2), (b1, b2) = sa(w), sb(w)
        da.append(b1 - a1)
        db_.append(b2 - a2)
    da, db_ = np.array(da), np.array(db_)
    p2 = lambda v: float(2 * min((v <= 0).mean(), (v >= 0).mean()))
    row = {"predictor": NAME, "n_common": int(common.sum()),
           "auroc_with_BRCA1": roc_auc_score(y, a), "auroc_no_BRCA1": roc_auc_score(y, b),
           "d_auroc": roc_auc_score(y, b) - roc_auc_score(y, a),
           "d_auroc_ci_low": interval(da)[0], "d_auroc_ci_high": interval(da)[1],
           "auroc_p": p2(da),
           "auprc_with_BRCA1": average_precision_score(y, a),
           "auprc_no_BRCA1": average_precision_score(y, b),
           "d_auprc": average_precision_score(y, b) - average_precision_score(y, a),
           "d_auprc_ci_low": interval(db_)[0], "d_auprc_ci_high": interval(db_)[1],
           "auprc_p": p2(db_)}
    pd.DataFrame([row]).to_csv(os.path.join(out, "paired_with_vs_without_BRCA1.csv"), index=False)
    print(pd.DataFrame([row]).T.to_string(header=False))


if __name__ == "__main__":
    main()
