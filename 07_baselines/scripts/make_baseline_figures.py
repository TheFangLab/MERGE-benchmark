"""Figures 7 (AUROC) and 8 (AUPRC): MERGE against established predictors.

One panel per variant class. Every predictor in a panel is evaluated on the same
common scored subset; points are means over the three chromosome-held-out folds
and whiskers are +-1 sample SD. Markers:

    MERGE                              red circle
    MERGE + AlphaGenome average        pale red diamond with a red outline
    not trained on clinical labels     blue circle   (tier A)
    clinical-label-trained             grey square   (tier B)

The dbNSFP copies of CADD and phyloP (`*_dbnsfp`) are a cross-check on the
directly obtained scores and are not drawn. Both non-coding AlphaGenome summaries
(raw and quantile maximum) are drawn.

Reads 07_baselines/results/baseline_comparison.csv; writes 06_figures/.

Usage: python make_baseline_figures.py [repo-root]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "04_modeling"))
import figstyle as fs                                   # noqa: E402

RESULTS = os.path.join(ROOT, "07_baselines", "results")
OUT = os.environ.get("MERGE_FIGURES", os.path.join(ROOT, "06_figures"))
TASKS = ["coding", "splicing", "noncoding", "nonsnv"]
TITLE = {"coding": "Coding missense SNVs", "splicing": "Splicing SNVs",
         "noncoding": "Noncoding SNVs", "nonsnv": "Short non-SNVs"}
GREY = "#7d7c76"
NAMES = {"MERGE + AlphaGenome average": "MERGE + AlphaGenome average",
         "AlphaGenome_raw_max": "AlphaGenome raw maximum",
         "AlphaGenome_quantile_max": "AlphaGenome quantile maximum",
         "phyloP100way": "phyloP 100-way", "Polyphen2_HVAR": "PolyPhen-2 HVAR",
         "frameshift_rule": "Frameshift rule", "ReMM_v031": "ReMM v0.3.1",
         "ReMM_v04": "ReMM v0.4"}
STYLE = {  # marker, face, edge
    "MERGE": ("o", fs.MERGE_ACCENT, fs.MERGE_ACCENT),
    "average": ("D", fs.MERGE_AVG_FILL, fs.MERGE_ACCENT),
    "A": ("o", fs.BLUE, fs.BLUE),
    "B": ("s", GREY, GREY),
}


def group(row):
    if row.predictor == "MERGE":
        return "MERGE"
    if row.predictor.startswith("MERGE"):
        return "average"
    return row.tier


def figure(cmp_, metric, stem):
    col, sd = f"{metric}_common", f"{metric}_sd_common"
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 12.5),
                             gridspec_kw={"height_ratios": [1.25, 1.0]})
    for ax, task, letter in zip(axes.ravel(), TASKS, "abcd"):
        d = cmp_[(cmp_.task == task) & cmp_[col].notna()].sort_values(col).reset_index(drop=True)
        y = np.arange(len(d))
        for yi, r in d.iterrows():
            m, face, edge = STYLE[group(r)]
            ax.errorbar(r[col], yi, xerr=r[sd], fmt="none", ecolor=edge, elinewidth=1.1,
                        capsize=2.5, zorder=3)
            ax.plot(r[col], yi, m, ms=7.5 if m != "D" else 7, mfc=face, mec=edge,
                    mew=1.4, zorder=4)
            ax.text(1.02, yi, f"{r[col]:.3f}", transform=ax.get_yaxis_transform(),
                    va="center", ha="left", fontsize=8, color=fs.INK_2,
                    fontfamily="monospace")
        ax.set_yticks(y)
        labels = [NAMES.get(p, p) for p in d.predictor]
        ax.set_yticklabels(labels, fontsize=8.5)
        for t, g in zip(ax.get_yticklabels(), d.apply(group, axis=1)):
            if g in ("MERGE", "average"):
                t.set_fontweight("bold"); t.set_color(fs.INK)
        lo = np.floor((d[col] - d[sd]).min() * 20 - 0.5) / 20
        ax.set_xlim(max(0.0, lo), 1.0)
        ax.set_ylim(-0.7, len(d) - 0.3)
        fs.grid_x(ax)
        fs.strip_spines(ax, keep=("bottom",))
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel(metric.upper(), fontsize=9.5)
        n, npos = int(d.n_common.iloc[0]), int(d.positives_common.iloc[0])
        fs.panel_title(ax, letter, f"{TITLE[task]}  (n = {n:,}; {npos / n:.1%} positive)")
    handles = [Line2D([], [], ls="none", marker=STYLE[k][0], mfc=STYLE[k][1], mec=STYLE[k][2],
                      mew=1.4, ms=7, label=lab)
               for k, lab in (("MERGE", "MERGE"), ("average", "MERGE + AlphaGenome average"),
                              ("A", "Not trained on clinical pathogenicity labels"),
                              ("B", "Trained on clinical pathogenicity labels"))]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.03, 1, 1), h_pad=3.0, w_pad=4.0)
    fs.save_figure(fig, OUT, stem)


def main():
    fs.apply_style()
    cmp_ = pd.read_csv(os.path.join(RESULTS, "baseline_comparison.csv"))
    cmp_ = cmp_[~cmp_.predictor.str.endswith("_dbnsfp")]
    figure(cmp_, "auroc", "fig7_baselines_auroc")
    figure(cmp_, "auprc", "fig8_baselines_auprc")


if __name__ == "__main__":
    main()
