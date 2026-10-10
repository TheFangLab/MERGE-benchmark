"""
Derive per-task interpretation cut-points for MERGE + AlphaGenome average.

MERGE + AlphaGenome average is a fixed 1:1 mean of the no-AlphaGenome ensemble
probability and an AlphaGenome percentile, so it is not on the same scale as the
ensemble and the ensemble's 0.8 / 0.6 / 0.4 / 0.2 bands do not transfer to it.

The cut-points here are call-rate matched on the chromosome-held-out OOF
predictions: for each ensemble cut c, the MERGE + AlphaGenome average cut is the quantile of
MERGE + AlphaGenome average at the fraction of variants scoring below c under the ensemble. That
keeps each band holding the same share of variants it holds for the ensemble,
and uses no labels, so no label information leaks into the thresholds. Labels are
used only to report the sensitivity/specificity each cut lands on, as a check.

The cut-points describe the deployed score, which is computed only for variants
that have an AlphaGenome score and against a reference of observed scores. They
are therefore calibrated on those variants, with the out-of-fold percentile
recomputed against the observed training-fold scores -- not on the
median-imputed evaluation column that `alphagenome_average.py` reports.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alphagenome_average import midrank_ecdf
from config import AGAVG as _AGAVG

AGAVG = Path(_AGAVG)
OUT = AGAVG

BASE_CUTS = [0.8, 0.6, 0.4, 0.2]          # ensemble bands, high to low
BAND_NAMES = ["Pathogenic", "Likely Pathogenic", "Uncertain Significance",
              "Likely Benign", "Benign"]               # as shown by the web server and merge-cli
# Offered for splicing and non-coding only; see 04_modeling/alphagenome_average.py.
TASKS = ["splicing", "noncoding"]


def rates(y, score, cut):
    call = score >= cut
    pos, neg = y == 1, y == 0
    return {
        "call_rate": float(call.mean()),
        "sensitivity": float(call[pos].mean()) if pos.any() else None,
        "specificity": float((~call)[neg].mean()) if neg.any() else None,
    }


def calibrate(task):
    df = pd.read_csv(AGAVG / f"{task}_alphagenome_average_oof.csv", low_memory=False)
    feature = str(df["alphagenome_feature"].iloc[0])
    # A deployed variant with no AlphaGenome score has no average, so it cannot
    # inform the cut-points either; the ensemble side is restricted to the same
    # rows, and the percentile is taken against observed training-fold scores.
    df = df[df["alphagenome_observed"] == 1].reset_index(drop=True)
    raw = df["alphagenome_raw"].to_numpy(dtype=float)
    folds = df["fold"].to_numpy()
    comp = np.full(len(df), np.nan)
    for k in np.unique(folds):
        te = folds == k
        comp[te] = midrank_ecdf(raw[te], raw[~te])
    y = df["label"].to_numpy(dtype=int)
    base = df["merge_no_alphagenome"].to_numpy(dtype=float)
    final = 0.5 * base + 0.5 * comp

    cuts, checks = [], []
    for c in BASE_CUTS:
        below = float((base < c).mean())                  # share the ensemble puts below c
        t = float(np.quantile(final, below)) if 0.0 < below < 1.0 else (0.0 if below <= 0 else 1.0)
        cuts.append(round(t, 4))
        checks.append({
            "ensemble_cut": c,
            "merge_alphagenome_average_cut": round(t, 4),
            "share_below_cut": round(below, 4),
            "ensemble": {k: (None if v is None else round(v, 4)) for k, v in rates(y, base, c).items()},
            "merge_alphagenome_average": {k: (None if v is None else round(v, 4)) for k, v in rates(y, final, t).items()},
        })

    # keep the cut-points strictly decreasing so the bands stay well-ordered
    for i in range(1, len(cuts)):
        if cuts[i] >= cuts[i - 1]:
            cuts[i] = round(cuts[i - 1] - 1e-4, 4)

    def band(scores, edges):
        return np.select([scores >= edges[0], scores >= edges[1],
                          scores >= edges[2], scores >= edges[3]],
                         BAND_NAMES[:4], BAND_NAMES[4])

    agree = float((band(base, BASE_CUTS) == band(final, cuts)).mean())
    naive = float((band(base, BASE_CUTS) == band(final, BASE_CUTS)).mean())
    return {
        "task": task,
        "alphagenome_feature": feature,
        "n_oof": int(len(df)),
        "pos_rate": round(float(y.mean()), 4),
        "cuts": cuts,
        "band_names": BAND_NAMES,
        "band_agreement_with_ensemble": round(agree, 4),
        "band_agreement_if_reusing_ensemble_cuts": round(naive, 4),
        "per_cut_check": checks,
        "source_oof": f"05_results_alphagenome_average/{task}_alphagenome_average_oof.csv",
        "component_transform": "mid-rank empirical CDF, no labels, no fitted model",
    }


if __name__ == "__main__":
    out = {t: calibrate(t) for t in TASKS}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "merge_final_thresholds.json").write_text(json.dumps(out, indent=2))
    for t, r in out.items():
        print(f"{t:10s} feat={r['alphagenome_feature']:32s} cuts={r['cuts']}  "
              f"band agreement {r['band_agreement_with_ensemble']:.3f} "
              f"(reusing ensemble cuts: {r['band_agreement_if_reusing_ensemble_cuts']:.3f})")
    print(f"wrote {OUT/'merge_final_thresholds.json'}")
