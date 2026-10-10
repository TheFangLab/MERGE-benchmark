"""
Paired comparisons of MERGE with single features on pooled out-of-fold predictions
(manuscript Table 3).

Two kinds of reference are compared, each named explicitly:

  candidate input         the feature with the highest mean outer-fold AUROC among
                          the inputs MERGE could select from (AlphaGenome excluded),
                          read from {task}_meta.json
  independent comparator  AlphaGenome, which MERGE never trains on: the splicing
                          score, and both non-coding maxima (raw and quantile)

For every comparison:
  - DeLong test of the AUROC difference (paired, Sun & Xu fast implementation)
  - 2,000 paired, class-stratified bootstrap resamples: 95% percentile intervals
    for AUROC, AUPRC and both differences

Each comparison draws from its own RandomState(42), so a result does not depend on
which other comparisons are run or in what order.

The bootstrap evaluates AUROC and average precision for each resample from
per-variant resample counts on a presorted score vector. That is the same
quantity as sklearn's roc_auc_score / average_precision_score on the resampled
rows (checked against sklearn at start-up), only faster.

Pooled metrics rank predictions across folds as well as within folds, so they
differ from the fold means reported elsewhere.

Outputs: <results>/stats/paired_tests.csv and paired_tests.json
Usage:   python stats_tests.py [--results DIR]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import scipy.stats
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS, RANDOM_STATE
from fast_metrics import WeightedScorer, check_against_sklearn

N_BOOT = 2000

# Independent AlphaGenome comparators, by task. Candidate-input references are
# not listed here: they are read from each task's meta.json.
ALPHAGENOME_REFERENCES = {
    "splicing": ["alphagenome_splicing"],
    "noncoding": ["alphagenome_quantile_score_max", "alphagenome_raw_score_max"],
}
TASKS = ["coding", "splicing", "noncoding", "nonsnv"]


# ---------------- fast DeLong (Sun & Xu 2014) -------------------------------
def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def fast_delong(preds_sorted, m):
    """preds_sorted: [k, n] with the m positives first. Returns aucs, cov."""
    n = preds_sorted.shape[1] - m
    pos = preds_sorted[:, :m]
    neg = preds_sorted[:, m:]
    k = preds_sorted.shape[0]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = compute_midrank(pos[r, :])
        ty[r, :] = compute_midrank(neg[r, :])
        tz[r, :] = compute_midrank(preds_sorted[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    return aucs, np.atleast_2d(cov)


def delong_test(y, p1, p2):
    """Two-sided DeLong p for AUC(p1) vs AUC(p2)."""
    order = np.argsort(-y)            # positives (label 1) first
    aucs, cov = fast_delong(np.vstack([p1, p2])[:, order], int(y.sum()))
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        z = 0.0 if aucs[0] == aucs[1] else np.inf * np.sign(aucs[0] - aucs[1])
    else:
        z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return float(z), float(2 * scipy.stats.norm.sf(abs(z)))


def bootstrap(y, pA, pB, seed=RANDOM_STATE, n_boot=N_BOOT):
    """Paired, class-stratified bootstrap of A (MERGE) and B (reference)."""
    rng = np.random.RandomState(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    A, B = WeightedScorer(pA, y), WeightedScorer(pB, y)
    draws = np.empty((n_boot, 4))
    for i in range(n_boot):
        bi = np.concatenate([rng.choice(pos, len(pos), replace=True),
                             rng.choice(neg, len(neg), replace=True)])
        w = np.bincount(bi, minlength=len(y)).astype(float)
        draws[i, :2] = A(w)
        draws[i, 2:] = B(w)

    def ci(v):
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    def p_two_sided(d):
        return float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))

    d_auroc, d_auprc = draws[:, 0] - draws[:, 2], draws[:, 1] - draws[:, 3]
    au_a, au_b = roc_auc_score(y, pA), roc_auc_score(y, pB)
    ap_a, ap_b = average_precision_score(y, pA), average_precision_score(y, pB)
    return {
        "merge_auroc": au_a, "merge_auroc_ci": ci(draws[:, 0]),
        "merge_auprc": ap_a, "merge_auprc_ci": ci(draws[:, 1]),
        "reference_auroc": au_b, "reference_auroc_ci": ci(draws[:, 2]),
        "reference_auprc": ap_b, "reference_auprc_ci": ci(draws[:, 3]),
        "delta_auroc": au_a - au_b, "delta_auroc_ci": ci(d_auroc),
        "delta_auroc_boot_p": p_two_sided(d_auroc),
        "delta_auprc": ap_a - ap_b, "delta_auprc_ci": ci(d_auprc),
        "delta_auprc_boot_p": p_two_sided(d_auprc),
    }


def comparisons(results):
    out = []
    for t in TASKS:
        meta = json.load(open(os.path.join(results, f"{t}_meta.json")))
        out.append((t, meta["best_single_nonalphagenome_feature"], "candidate input"))
    for t, feats in ALPHAGENOME_REFERENCES.items():
        out.extend((t, f, "independent comparator") for f in feats)
    return out


def run(results):
    rows = []
    for task, ref, role in comparisons(results):
        meta = json.load(open(os.path.join(results, f"{task}_meta.json")))
        strat = meta["best_strategy"]
        mo = pd.read_csv(os.path.join(results, f"{task}_merge_oof.csv"), low_memory=False)
        sfo = pd.read_csv(os.path.join(results, f"{task}_singlefeature_oof.csv"),
                          low_memory=False)
        # the two OOF tables are written row for row from the same dataset
        assert (mo["pos"].values == sfo["pos"].values).all()
        d = pd.DataFrame({"y": mo["label"].values, "merge": mo[strat].values,
                          "ref": sfo[ref].values}).dropna()
        y = d["y"].values.astype(int)
        check_against_sklearn(y, d["merge"].values)
        bs = bootstrap(y, d["merge"].values, d["ref"].values)
        z, p = delong_test(y, d["merge"].values, d["ref"].values)
        row = {"task": task, "comparison": role, "reference": ref,
               "merge_strategy": strat, "n": int(len(y)), "positives": int(y.sum()),
               "delong_z": z, "delong_p": p}
        row.update(bs)
        rows.append(row)
        print(f"[{task}] MERGE - {ref}: dAUROC {bs['delta_auroc']:+.4f} "
              f"[{bs['delta_auroc_ci'][0]:+.4f}, {bs['delta_auroc_ci'][1]:+.4f}] "
              f"DeLong p={p:.2e}  dAUPRC {bs['delta_auprc']:+.4f} "
              f"[{bs['delta_auprc_ci'][0]:+.4f}, {bs['delta_auprc_ci'][1]:+.4f}]",
              flush=True)

    outdir = os.path.join(results, "stats")
    os.makedirs(outdir, exist_ok=True)
    json.dump(rows, open(os.path.join(outdir, "paired_tests.json"), "w"), indent=1)
    flat = []
    for r in rows:
        f = {k: v for k, v in r.items() if not isinstance(v, list)}
        for k, v in r.items():
            if isinstance(v, list):
                f[k + "_lo"], f[k + "_hi"] = v
        flat.append(f)
    pd.DataFrame(flat).to_csv(os.path.join(outdir, "paired_tests.csv"), index=False)
    print("wrote", os.path.join(outdir, "paired_tests.csv"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=RESULTS)
    run(ap.parse_args().results)
