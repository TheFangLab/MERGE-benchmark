"""Probability calibration and chromosome-cluster bootstrap for the four MERGE tasks.

Produces manuscript Table 5 (Brier score, expected calibration error, reliability
curves) and the chromosome-cluster bootstrap intervals for MERGE minus the strongest
single feature quoted in the Results (coding: AlphaMissense; non-SNV: Evo 2 7B).

Inputs  : results/{task}_merge_oof.csv, results/{task}_singlefeature_oof.csv,
          results/{task}_meta.json
Outputs : results/calibration/calibration.json, results/stats_cluster/cluster_bootstrap.json

Usage:
    python calibration_and_cluster_bootstrap.py [--results DIR] [--out-suffix ""]
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS

TASKS = ["coding", "splicing", "noncoding", "nonsnv"]
N_BOOT = 2000
N_BINS = 10
SEED = 42


def load_task(results: str, task: str):
    meta = json.load(open(os.path.join(results, f"{task}_meta.json")))
    strategy = meta["best_strategy"]
    feature = meta["best_single_feature"]
    merge = pd.read_csv(os.path.join(results, f"{task}_merge_oof.csv"), low_memory=False)
    single = pd.read_csv(os.path.join(results, f"{task}_singlefeature_oof.csv"), low_memory=False)

    # The two out-of-fold tables are written row-for-row by pipeline.py; align by row.
    df = pd.DataFrame(
        {
            "chr": merge["chr"].astype(str).str.replace("chr", "", regex=False),
            "label": merge["label"].astype(int),
            "merge": merge[strategy].astype(float),
            "single": single[feature].astype(float),
        }
    ).dropna().reset_index(drop=True)
    return meta, strategy, feature, df


# ------------------------------------------------------------------------------------
# Calibration (manuscript Table 5)
# ------------------------------------------------------------------------------------
def calibration(y: np.ndarray, p: np.ndarray) -> Dict[str, object]:
    base = float(y.mean())
    brier = float(np.mean((p - y) ** 2))
    brier_baseline = float(np.mean((base - y) ** 2))

    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, N_BINS - 1)

    ece = 0.0
    mce = 0.0
    reliability: List[List[float]] = []
    for b in range(N_BINS):
        mask = idx == b
        count = int(mask.sum())
        centre = float((edges[b] + edges[b + 1]) / 2)
        if count == 0:
            reliability.append([centre, 0.0, 0.0, 0])
            continue
        mean_pred = float(p[mask].mean())
        frac_pos = float(y[mask].mean())
        gap = abs(mean_pred - frac_pos)
        ece += gap * count / len(y)
        mce = max(mce, gap)
        reliability.append([centre, round(mean_pred, 4), round(frac_pos, 4), count])

    return {
        "n": int(len(y)),
        "base_rate": round(base, 4),
        "brier": round(brier, 4),
        "brier_baseline": round(brier_baseline, 4),
        "ece": round(ece, 4),
        "mce": round(mce, 4),
        "reliability": reliability,
    }


# ------------------------------------------------------------------------------------
# Chromosome-cluster bootstrap (manuscript Section 3.7)
# ------------------------------------------------------------------------------------
def cluster_bootstrap(df: pd.DataFrame) -> Dict[str, object]:
    y = df["label"].to_numpy()
    pm = df["merge"].to_numpy()
    ps = df["single"].to_numpy()
    if roc_auc_score(y, ps) < 0.5:  # orient the single feature like the logistic fit
        ps = -ps

    groups = {c: np.asarray(i) for c, i in df.groupby("chr").indices.items()}
    chroms = sorted(groups)

    rng = np.random.RandomState(SEED)
    d_auroc, d_auprc, b_auroc_m, b_auroc_s, b_auprc_m = [], [], [], [], []
    for _ in range(N_BOOT):
        picked = rng.choice(len(chroms), len(chroms), replace=True)
        sel = np.concatenate([groups[chroms[k]] for k in picked])
        yb = y[sel]
        if yb.min() == yb.max():
            continue
        am = roc_auc_score(yb, pm[sel])
        asg = roc_auc_score(yb, ps[sel])
        pm_ap = average_precision_score(yb, pm[sel])
        ps_ap = average_precision_score(yb, ps[sel])
        b_auroc_m.append(am)
        b_auroc_s.append(asg)
        b_auprc_m.append(pm_ap)
        d_auroc.append(am - asg)
        d_auprc.append(pm_ap - ps_ap)

    def ci(v):
        return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]

    d_auroc = np.asarray(d_auroc)
    d_auprc = np.asarray(d_auprc)
    return {
        "n": int(len(y)),
        "n_chr": len(chroms),
        "merge_auroc": round(float(roc_auc_score(y, pm)), 4),
        "merge_auroc_ci_cluster": ci(b_auroc_m),
        "single_auroc": round(float(roc_auc_score(y, ps)), 4),
        "single_auroc_ci_cluster": ci(b_auroc_s),
        "delta_auroc": round(float(roc_auc_score(y, pm) - roc_auc_score(y, ps)), 4),
        "delta_auroc_ci_cluster": ci(d_auroc),
        "delta_auroc_p_cluster": round(float(2 * min((d_auroc <= 0).mean(), (d_auroc >= 0).mean())), 4),
        "merge_auprc": round(float(average_precision_score(y, pm)), 4),
        "merge_auprc_ci_cluster": ci(b_auprc_m),
        "single_auprc": round(float(average_precision_score(y, ps)), 4),
        "delta_auprc": round(float(average_precision_score(y, pm) - average_precision_score(y, ps)), 4),
        "delta_auprc_ci_cluster": ci(d_auprc),
        "delta_auprc_p_cluster": round(float(2 * min((d_auprc <= 0).mean(), (d_auprc >= 0).mean())), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=RESULTS)
    parser.add_argument("--out-suffix", default="", help="Appended to the output file stems")
    args = parser.parse_args()

    cal_dir = os.path.join(args.results, "calibration")
    clu_dir = os.path.join(args.results, "stats_cluster")
    os.makedirs(cal_dir, exist_ok=True)
    os.makedirs(clu_dir, exist_ok=True)

    calibrations: List[Dict[str, object]] = []
    clusters: List[Dict[str, object]] = []

    for task in TASKS:
        print(f"[{task}]", flush=True)
        meta, strategy, feature, df = load_task(args.results, task)
        cal = {"task": task, "strategy": strategy}
        cal.update(calibration(df["label"].to_numpy(), df["merge"].to_numpy()))
        calibrations.append(cal)
        print(f"  calibration: n={cal['n']} base_rate={cal['base_rate']} "
              f"brier={cal['brier']} ece={cal['ece']}", flush=True)

        clu = {"task": task}
        clu.update(cluster_bootstrap(df))
        clusters.append(clu)
        print(f"  cluster bootstrap: n={clu['n']} n_chr={clu['n_chr']} "
              f"dAUROC={clu['delta_auroc']} {clu['delta_auroc_ci_cluster']}", flush=True)

    suffix = args.out_suffix
    with open(os.path.join(cal_dir, f"calibration{suffix}.json"), "w") as fh:
        json.dump(calibrations, fh, indent=1)
    with open(os.path.join(clu_dir, f"cluster_bootstrap{suffix}.json"), "w") as fh:
        json.dump(clusters, fh, indent=1)
    print("SAVED")


if __name__ == "__main__":
    main()
