"""Figure 9: AUROC and AUPRC on the BRCA1 SGE common subset.

MERGE, the 17 established predictors and the 25 individual benchmark features,
each with its 95% percentile interval from 2,000 paired, class-stratified
bootstrap resamples. Open squares and daggers mark clinical-label-trained
comparators; dashed lines mark random performance.

Reads results/sge_external_validation_metrics.csv; writes 06_figures/.

Usage: python 06_make_sge_figure.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
REPO = os.path.dirname(EXP)
sys.path.insert(0, os.path.join(REPO, "04_modeling"))
import figstyle as fs                                   # noqa: E402

OUT = os.environ.get("MERGE_FIGURES", os.path.join(REPO, "06_figures"))
DISPLAY = {
    "MERGE without AlphaGenome": "MERGE",
    "AlphaMissense_score": "AlphaMissense", "ESM1b_score": "ESM-1b",
    "GPN_MSA_score": "GPN-MSA", "carbon_score": "Carbon-3B",
    "evo2_7b_score": "Evo 2 7B", "evo2_7b_base_score": "Evo 2 7B-base",
    "evo2_1b_base_score": "Evo 2 1B-base",
    "generator_score": "GENERATOR", "generator_v2_score": "GENERATOR-v2",
    "NT_score": "Nucleotide Transformer", "nt_v2_score": "Nucleotide Transformer v2",
    "heyenadna_score": "HyenaDNA",
    "alphagenome_raw_score_max": "AlphaGenome raw maximum",
    "alphagenome_raw_score_min": "AlphaGenome raw minimum",
    "alphagenome_raw_score_mean": "AlphaGenome raw mean",
    "alphagenome_quantile_score_max": "AlphaGenome quantile maximum",
    "alphagenome_quantile_score_min": "AlphaGenome quantile minimum",
    "alphagenome_quantile_score_mean": "AlphaGenome quantile mean",
    "enformer_absmean": "Enformer absolute mean", "enformer_mean": "Enformer mean",
    "enformer_std": "Enformer SD", "enformer_max_abs": "Enformer maximum |effect|",
    "enformer_top5_absmean": "Enformer top-5 absolute mean",
    "enformer_pos_sum": "Enformer positive sum", "enformer_neg_sum": "Enformer negative sum",
}
STYLE = {  # marker, edge colour
    "MERGE": ("o", fs.MERGE_ACCENT),
    "comparator": ("s", fs.BLUE),
    "feature": ("^", fs.ORANGE),
}


def draw(ax, d, metric, chance):
    lo_c, hi_c = f"{metric}_ci_low", f"{metric}_ci_high"
    d = d.sort_values([metric, "label"], ascending=[False, True]).reset_index(drop=True)
    lo = max(0.0, min(float(d[lo_c].min()) - 0.04, chance - 0.035))
    ax.set_xlim(lo, 1.0)
    ax.set_ylim(-0.8, len(d) - 0.2)
    ax.invert_yaxis()
    ax.axvline(chance, color=fs.MUTED, lw=0.9, ls=(0, (3, 3)), zorder=1)
    for yi, r in d.iterrows():
        marker, colour = STYLE["MERGE" if r.kind == "MERGE" else
                               "comparator" if r.kind.startswith("comparator") else "feature"]
        face = "white" if r.kind == "comparator (label-trained)" else colour
        ax.errorbar(r[metric], yi, xerr=[[r[metric] - r[lo_c]], [r[hi_c] - r[metric]]],
                    fmt=marker, ms=5.4, mfc=face, mec=colour, mew=1.1, ecolor=colour,
                    elinewidth=1.0, capsize=2.2, zorder=3)
        ax.text(1.015, yi, f"{r[metric]:.3f}", transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=7.2, color=fs.INK_2, fontfamily="monospace")
    ax.set_yticks(np.arange(len(d)))
    ax.set_yticklabels(d["label"], fontsize=7.6)
    ax.get_yticklabels()[int(np.flatnonzero(d.kind == "MERGE")[0])].set_fontweight("bold")
    ax.tick_params(axis="y", length=0, pad=4)
    ax.set_xticks(np.arange(np.ceil(lo * 10) / 10, 1.0001, 0.1))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    fs.grid_x(ax)
    fs.strip_spines(ax, keep=("bottom",))
    ax.set_xlabel(metric.upper(), fontsize=9.5)


def main():
    fs.apply_style()
    d = pd.read_csv(os.path.join(EXP, "results", "sge_external_validation_metrics.csv"))
    pred = pd.read_csv(os.path.join(EXP, "results", "sge_merge_predictions.csv"))
    prevalence = float(pred.loc[pred.in_common_subset, "label"].mean())
    d["label"] = d["predictor"].map(lambda p: DISPLAY.get(p, p))
    d.loc[d.kind == "comparator (label-trained)", "label"] += " †"

    fig, axes = plt.subplots(1, 2, figsize=(15, 0.27 * len(d) + 2.0))
    draw(axes[0], d, "auroc", 0.5)
    draw(axes[1], d, "auprc", prevalence)
    handles = [
        Line2D([], [], ls="none", marker="o", mfc=fs.MERGE_ACCENT, mec=fs.MERGE_ACCENT,
               label="MERGE"),
        Line2D([], [], ls="none", marker="s", mfc=fs.BLUE, mec=fs.BLUE,
               label="Established predictor"),
        Line2D([], [], ls="none", marker="s", mfc="white", mec=fs.BLUE,
               label="Established predictor, clinical-label-trained (†)"),
        Line2D([], [], ls="none", marker="^", mfc=fs.ORANGE, mec=fs.ORANGE,
               label="Individual model-derived feature"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.03, 1, 1), w_pad=6.0)
    fs.save_figure(fig, OUT, "fig9_sge_external_validation")


if __name__ == "__main__":
    main()
