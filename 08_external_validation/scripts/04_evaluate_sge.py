"""
Evaluate the frozen MERGE-Coding model and every comparator on the BRCA1 SGE cohort
(manuscript Table 9 and Figure 9).

MERGE is the deployed coding bundle, 05_results/trained_models/
MERGE_noAlphaGenome_coding.pkl: fitted on all 4,289 coding training variants with the
benchmark-selected strategy, frozen before it saw this cohort. No SGE label is used
for feature selection, orientation, preprocessing or fitting.

Predictors (all oriented so that larger = more damaging):
  * MERGE
  * the 25 individual benchmark features. AlphaGenome columns use the direction
    stated in 04_modeling/singlefeature_benchmark.py; every other column takes the
    sign of its AUROC on the coding *training* table.
  * 17 established predictors from dbNSFP 5.3a (model_outputs/sge_dbnsfp_scores.csv;
    SIFT4G and PROVEAN are already negated there).

All predictors are compared on the variants that every one of them scores (the
common subset, n = 1,421). Intervals and two-sided P values come from 2,000 paired,
class-stratified variant-level bootstrap resamples (numpy default_rng(42)); the same
resamples are used for every predictor.

Outputs (results/):
  sge_merge_predictions.csv           per-variant MERGE score and common-subset flag
  sge_external_validation_metrics.csv AUROC / AUPRC with 95% intervals, per predictor
  sge_paired_bootstrap_deltas.csv     MERGE minus each predictor
  sge_modeling_meta.json

Usage: python 04_evaluate_sge.py
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
REPO = os.path.dirname(EXP)
sys.path.insert(0, os.path.join(REPO, "04_modeling"))
from fast_metrics import WeightedScorer, check_against_sklearn   # noqa: E402
from singlefeature_benchmark import ORIENTATION                   # noqa: E402

N_BOOT = 2000
SEED = 42
MERGE_COL = "MERGE without AlphaGenome"
BUNDLE = os.path.join(REPO, "05_results", "trained_models", "MERGE_noAlphaGenome_coding.pkl")
KEY = ["chr", "pos", "ref", "alt"]
ID = ["variant_id", "chr", "pos", "ref", "alt", "label"]
COMPARATORS = ["SIFT4G", "PolyPhen-2 HVAR", "MutationAssessor", "PROVEAN",
               "PrimateAI", "VEST4", "MetaRNN", "M-CAP", "REVEL", "BayesDel",
               "ClinPred", "VARITY", "CADD", "GERP++", "phyloP 100-way",
               "phyloP 470-way", "MutFormer"]
# Trained on ClinVar and/or HGMD labels.
LABEL_TRAINED = {"VEST4", "MetaRNN", "M-CAP", "REVEL", "BayesDel", "ClinPred",
                 "VARITY", "MutFormer"}


def feature_directions(features):
    """Sign of each feature, fixed without the SGE labels."""
    train = pd.read_csv(os.path.join(REPO, "03_datasets", "coding_features.csv"),
                        low_memory=False)
    y = train["label"].to_numpy(int)
    out = {}
    for c in features:
        if c in ORIENTATION:
            out[c] = ORIENTATION[c]
            continue
        x = pd.to_numeric(train[c], errors="coerce").to_numpy(float)
        ok = np.isfinite(x)
        out[c] = 1 if roc_auc_score(y[ok], x[ok]) >= 0.5 else -1
    return out


def main():
    sge = pd.read_csv(os.path.join(EXP, "features", "sge_features.csv"), low_memory=False)
    db = pd.read_csv(os.path.join(EXP, "model_outputs", "sge_dbnsfp_scores.csv"),
                     low_memory=False)
    for d in (sge, db):
        d["chr"] = d["chr"].astype(str)
    sge = sge.merge(db[KEY + COMPARATORS], on=KEY, how="left", validate="one_to_one")
    y_all = sge["label"].to_numpy(int)
    features = [c for c in sge.columns if c not in ID + COMPARATORS]

    bundle = joblib.load(BUNDLE)
    if any(f.startswith("alphagenome") for f in bundle["features"]):
        raise SystemExit("the coding bundle contains an alphagenome_* feature; refusing")
    merge = bundle["pipeline"].predict_proba(sge[bundle["features"]].to_numpy())[:, 1]

    directions = feature_directions(features)
    preds = {MERGE_COL: (merge, "MERGE")}
    for c in features:
        preds[c] = (directions[c] * pd.to_numeric(sge[c], errors="coerce").to_numpy(float),
                    "feature")
    for c in COMPARATORS:
        preds[c] = (pd.to_numeric(sge[c], errors="coerce").to_numpy(float),
                    "comparator (label-trained)" if c in LABEL_TRAINED else "comparator")

    common = np.ones(len(sge), bool)
    for p, _ in preds.values():
        common &= np.isfinite(p)
    y = y_all[common]
    print(f"SGE cohort n={len(sge)} ({y_all.sum()} LOF); common subset n={common.sum()} "
          f"({y.sum()} LOF)")

    rng = np.random.default_rng(SEED)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    weights = [np.bincount(np.concatenate([rng.choice(pos, pos.size, True),
                                           rng.choice(neg, neg.size, True)]),
                           minlength=len(y)) for _ in range(N_BOOT)]
    check_against_sklearn(y, merge[common])

    draws, rows = {}, []
    for name, (p, kind) in preds.items():
        pc = p[common]
        sc = WeightedScorer(pc, y)
        d = np.array([sc(w) for w in weights])
        draws[name] = (pc, d)
        rows.append({"predictor": name, "kind": kind, "n_common": int(common.sum()),
                     "coverage_full": round(float(np.isfinite(p).mean()), 6),
                     "auroc": roc_auc_score(y, pc),
                     "auroc_ci_low": np.percentile(d[:, 0], 2.5),
                     "auroc_ci_high": np.percentile(d[:, 0], 97.5),
                     "auprc": average_precision_score(y, pc),
                     "auprc_ci_low": np.percentile(d[:, 1], 2.5),
                     "auprc_ci_high": np.percentile(d[:, 1], 97.5)})
    res = pd.DataFrame(rows).sort_values("auroc", ascending=False).reset_index(drop=True)

    base, bd = draws[MERGE_COL]
    drows = []
    for name, (pc, d) in draws.items():
        if name == MERGE_COL:
            continue
        dau, dap = bd[:, 0] - d[:, 0], bd[:, 1] - d[:, 1]
        drows.append({
            "comparison": f"{MERGE_COL} - {name}",
            "delta_auroc": roc_auc_score(y, base) - roc_auc_score(y, pc),
            "delta_auroc_ci_low": np.percentile(dau, 2.5),
            "delta_auroc_ci_high": np.percentile(dau, 97.5),
            "p_auroc": float(2 * min((dau <= 0).mean(), (dau >= 0).mean())),
            "delta_auprc": average_precision_score(y, base) - average_precision_score(y, pc),
            "delta_auprc_ci_low": np.percentile(dap, 2.5),
            "delta_auprc_ci_high": np.percentile(dap, 97.5),
            "p_auprc": float(2 * min((dap <= 0).mean(), (dap >= 0).mean()))})
    dd = pd.DataFrame(drows).sort_values("delta_auroc").reset_index(drop=True)

    out = os.path.join(EXP, "results")
    os.makedirs(out, exist_ok=True)
    scored = sge[ID].copy()
    scored[MERGE_COL] = merge
    scored["in_common_subset"] = common
    scored.to_csv(os.path.join(out, "sge_merge_predictions.csv"), index=False)
    res.to_csv(os.path.join(out, "sge_external_validation_metrics.csv"), index=False)
    dd.to_csv(os.path.join(out, "sge_paired_bootstrap_deltas.csv"), index=False)
    json.dump({"cohort_n": int(len(sge)), "positives": int(y_all.sum()),
               "common_subset": int(common.sum()), "common_subset_positives": int(y.sum()),
               "n_bootstrap": N_BOOT, "bootstrap": "paired, class-stratified, "
               "variant-level; numpy default_rng(42)",
               "merge_bundle": os.path.relpath(BUNDLE, REPO),
               "merge_features": bundle["features"], "merge_algorithm": bundle["algorithm"],
               "feature_directions": directions},
              open(os.path.join(out, "sge_modeling_meta.json"), "w"), indent=1)
    print(res[["predictor", "kind", "auroc", "auprc"]].head(12).to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
