"""Chromosome-clustered bootstrap for MERGE minus a baseline.

Matches the resampling already used in 04_modeling/calibration_and_cluster_bootstrap.py:
chromosomes are the resampling unit, because variants on one chromosome are not
independent of each other and a naive per-variant bootstrap would give intervals
that are far too narrow.

Reports the difference in AUROC and AUPRC with a percentile interval. An
interval that spans zero means the ranking of the two methods is not resolved by
this dataset -- which is a result worth reporting, not a failure.

Appends its result to 07_baselines/results/significance.csv so the figures can
read the intervals rather than having them retyped.

Two MERGE anchors are supported. `merge` is the ensemble, which is trained
without AlphaGenome; `agavg` is the optional MERGE + AlphaGenome average, offered
on splicing and non-coding variants. Both are reported for those two variant
types, because on them AlphaGenome's own score outranks the ensemble and quoting
only one of the two anchors would misrepresent the comparison in one direction or
the other.

Usage: python merge_vs_baseline_ci.py --table8
       python merge_vs_baseline_ci.py <task> <baseline-tool> [n-boot] [merge|agavg]
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import (DATASETS, ROOT, SCORES, load_dataset, load_scores,
                      merge_oof)

sys.path.insert(0, os.path.join(ROOT, "04_modeling"))
from fast_metrics import WeightedScorer, check_against_sklearn


ANCHOR_DIR = {"merge": "05_results", "agavg": "05_results_alphagenome_average"}
ANCHOR_NAME = {"merge": "MERGE", "agavg": "MERGE + AlphaGenome average"}


def paired(task, tool, n_boot=2000, anchor="merge"):
    df = load_dataset(task)
    df = df[df["fold"] != -1].reset_index(drop=True)
    y = df["label"].to_numpy().astype(int)
    chrom = df["chr"].to_numpy()
    keys = list(zip(df["chr"], df["pos"], df["ref"], df["alt"]))

    merge, _ = merge_oof(task, ANCHOR_DIR[anchor])
    if merge is None:
        print(f"{task}: {ANCHOR_NAME[anchor]} not available, skipped")
        return
    lookup = load_scores(task)[tool]
    base = np.array([lookup.get(k, np.nan) for k in keys], dtype=float)

    ok = ~np.isnan(base) & ~np.isnan(merge)
    y, chrom, merge, base = y[ok], chrom[ok], merge[ok], base[ok]
    print(f"{task}: {ANCHOR_NAME[anchor]} vs {tool} on {ok.sum()} shared "
          f"variants ({y.sum()} positive)")

    idx_by_chrom = {c: np.flatnonzero(chrom == c) for c in np.unique(chrom)}
    chroms = np.array(list(idx_by_chrom))

    # A resample of whole chromosomes is a per-variant count vector: every
    # variant on a chromosome drawn k times carries weight k.
    check_against_sklearn(y, merge)
    sm, sb = WeightedScorer(merge, y), WeightedScorer(base, y)
    chrom_code = np.searchsorted(chroms, chrom)

    def delta(w):
        if (w * y).sum() == 0 or (w * (1 - y)).sum() == 0:
            return None
        (am, pm), (ab, pb) = sm(w), sb(w)
        return am - ab, pm - pb

    point = (roc_auc_score(y, merge) - roc_auc_score(y, base),
             average_precision_score(y, merge) - average_precision_score(y, base))
    rng = np.random.default_rng(42)          # one stream per comparison
    d_auroc, d_auprc = [], []
    for _ in range(n_boot):
        pick = rng.choice(chroms, size=len(chroms), replace=True)
        times = np.bincount(np.searchsorted(chroms, pick), minlength=len(chroms))
        d = delta(times[chrom_code].astype(float))
        if d is not None:
            d_auroc.append(d[0]); d_auprc.append(d[1])

    row = {"task": task, "anchor": ANCHOR_NAME[anchor], "comparator": tool,
           "n": int(ok.sum()), "positives": int(y.sum())}
    for name, obs, draws in (("auroc", point[0], d_auroc),
                             ("auprc", point[1], d_auprc)):
        lo, hi = np.percentile(draws, [2.5, 97.5])
        resolved = lo > 0 or hi < 0
        row.update({f"d_{name}": round(obs, 4), f"{name}_lo": round(lo, 4),
                    f"{name}_hi": round(hi, 4), f"{name}_resolved": resolved})
        print(f"  d{name.upper()} = {obs:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
              f"{'excludes 0' if resolved else 'SPANS 0 -- not resolved'}")

    out = os.path.join(ROOT, "07_baselines", "results", "significance.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    new = pd.DataFrame([row])
    if os.path.exists(out):
        old = pd.read_csv(out)
        if "anchor" not in old.columns:
            old["anchor"] = ANCHOR_NAME["merge"]
        old = old[~((old.task == task) & (old.comparator == tool)
                    & (old.anchor == ANCHOR_NAME[anchor]))]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(out, index=False)
    print(f"  -> {out}")


# The comparisons reported in manuscript Table 8, in table order.
TABLE8 = [
    ("coding", "CADD", "merge"),
    ("splicing", "SpliceAI", "merge"),
    ("splicing", "Pangolin", "merge"),
    ("splicing", "AlphaGenome", "merge"),
    ("noncoding", "CADD", "merge"),
    ("noncoding", "AlphaGenome_quantile_max", "merge"),
    ("noncoding", "AlphaGenome_raw_max", "merge"),
    ("nonsnv", "CADD", "merge"),
    ("splicing", "SpliceAI", "agavg"),
    ("splicing", "Pangolin", "agavg"),
    ("splicing", "AlphaGenome", "agavg"),
    ("noncoding", "CADD", "agavg"),
    ("noncoding", "AlphaGenome_quantile_max", "agavg"),
    ("noncoding", "AlphaGenome_raw_max", "agavg"),
]


if __name__ == "__main__":
    if sys.argv[1:] == ["--table8"]:
        out = os.path.join(ROOT, "07_baselines", "results", "significance.csv")
        if os.path.exists(out):
            os.remove(out)
        for task, tool, anchor in TABLE8:
            paired(task, tool, 2000, anchor)
    else:
        paired(sys.argv[1], sys.argv[2],
               int(sys.argv[3]) if len(sys.argv) > 3 else 2000,
               sys.argv[4] if len(sys.argv) > 4 else "merge")
