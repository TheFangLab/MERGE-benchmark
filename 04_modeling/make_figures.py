"""Generate the manuscript figures from results/ tables.

Presentation lives in figstyle.py; the numbers come straight from results/ and are
not touched here.

Produces Figure 2 (AUROC), Figure 3 (AUPRC) and Figure 5 (SHAP); Figures 4 and 6
are in make_figures2.py, 7 and 8 in 07_baselines/scripts/make_baseline_figures.py,
and 9 in 08_external_validation/scripts/06_make_sge_figure.py. Each figure is written twice: PNG for on-screen reading and
LZW-compressed TIFF at 300 dpi for submission.

Figures carry no titles: the caption is supplied by the manuscript. Panel letters
and axis labels stay, so each panel is self-describing.
"""
import os, sys, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import AGAVG, FIGURES, RESULTS
import figstyle as fs

fs.apply_style()

TASK_ORDER = ["coding", "splicing", "noncoding", "nonsnv"]
TASK_LABEL = {"coding": "Coding SNVs", "splicing": "Splicing SNVs",
              "noncoding": "Non-coding SNVs", "nonsnv": "Non-SNV variants"}
PANEL = ["a", "b", "c", "d"]



def meta(task):
    with open(f"{RESULTS}/{task}_meta.json") as f:
        return json.load(f)


def save(fig, stem):
    fs.save_figure(fig, FIGURES, stem)


# ---- Figure 2 / 3 : single-feature vs best ensemble ------------------------
def ag_average(task, metric):
    """The optional MERGE + AlphaGenome average, or None where it is not offered.

    Offered on splicing and non-coding variants only; evaluated on every variant,
    with missing AlphaGenome scores set to the training-fold median.
    """
    path = f"{AGAVG}/alphagenome_average_summary.csv"
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path)
    row = d[(d["task"] == task) & (d["offered"] == "yes")
            & (d["predictor"] == "MERGE + AlphaGenome average")]
    if row.empty:
        return None
    return float(row.iloc[0]["auroc" if metric == "auroc" else "auprc"])


def fig_single_vs_ensemble(metric, stem):
    """Every predictor for the task on one ranked axis.

    MERGE and the optional MERGE + AlphaGenome average sit at their true rank
    rather than being pinned to the top: on splicing and non-coding variants
    AlphaGenome's own score outranks the ensemble, and a figure that hid that
    would be reporting a different result than the tables.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 13))
    drawn = []
    for i, task in enumerate(TASK_ORDER):
        ax = axes[i // 2][i % 2]
        col = "auroc_mean" if metric == "auroc" else "auprc_mean"
        sf = pd.read_csv(f"{RESULTS}/{task}_singlefeature.csv").sort_values(col, ascending=True)
        m = meta(task)
        avg = ag_average(task, metric)

        # (value, label, facecolor, edgecolor, emphasis) for every row
        ramp = fs.rank_colors(len(sf))
        entries = [(float(v), fs.pretty(f), c, "none", False)
                   for (f, v), c in zip(sf[["feature", col]].itertuples(index=False), ramp)]
        entries.append((float(m["merge_auroc"] if metric == "auroc" else m["merge_auprc"]),
                        fs.merge_label(m["best_strategy"]), fs.MERGE_ACCENT, "none", True))
        if avg is not None:
            entries.append((float(avg), fs.MERGE_AVG_LABEL,
                            fs.MERGE_AVG_FILL, fs.MERGE_ACCENT, True))
        entries.sort(key=lambda e: e[0])

        vals = [e[0] for e in entries]
        names = [e[1] for e in entries]
        colors = [e[2] for e in entries]
        edges = [e[3] for e in entries]
        emph = [e[4] for e in entries]

        fs.grid_x(ax)
        bars = ax.barh(range(len(vals)), vals, height=0.68, color=colors, zorder=3,
                       edgecolor=edges, linewidth=1.2)
        drawn.append((ax, bars))
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(names, fontsize=8)
        for yi, e in enumerate(emph):
            if e:
                ax.get_yticklabels()[yi].set_fontweight("bold")
                ax.get_yticklabels()[yi].set_color(fs.INK)

        ax.set_xlim(0, 1.14)
        ax.set_xticks(np.arange(0, 1.01, 0.2))
        ax.set_ylim(-0.7, len(vals) - 0.3)
        ax.set_xlabel(metric.upper(), fontsize=9)
        fs.strip_spines(ax, keep=("bottom",))
        ax.tick_params(axis="y", length=0)

        # direct-label the MERGE rows and whatever is top of the axis
        labelled = {yi for yi, e in enumerate(emph) if e} | {len(vals) - 1}
        for yi in sorted(labelled):
            ax.text(vals[yi] + 0.015, yi, f"{vals[yi]:.3f}", va="center", fontsize=8,
                    color=fs.INK if emph[yi] else fs.INK_2,
                    fontweight="bold" if emph[yi] else "normal")

        if metric == "auprc":
            ax.axvline(m["pos_rate"], ls=(0, (4, 3)), c=fs.MUTED, lw=1, zorder=2)
            ax.text(m["pos_rate"], -0.62, "  no-skill", fontsize=7,
                    color=fs.MUTED, va="bottom", ha="left")

        fs.panel_title(ax, PANEL[i], f"{TASK_LABEL[task]}  (n = {m['n']:,})")

    fig.tight_layout()
    for ax, bars in drawn:                      # rounded data-ends, square at the baseline
        fs.round_bar_ends(ax, bars, "h")
    save(fig, stem)


# ---- Figure 5 : SHAP beeswarm ---------------------------------------------
def fig_shap():
    fig, axes = plt.subplots(2, 2, figsize=(15, 13))
    sc = None
    for i, task in enumerate(TASK_ORDER):
        ax = axes[i // 2][i % 2]
        p = f"{RESULTS}/{task}_shap_beeswarm.csv"
        if not os.path.exists(p):
            ax.set_visible(False); continue
        d = pd.read_csv(p)
        order = (d.groupby("feature")["shap"].apply(lambda s: s.abs().mean())
                 .sort_values(ascending=True))
        feats = list(order.index)
        rng = np.random.default_rng(42)
        for yi, fn in enumerate(feats):
            sub = d[d["feature"] == fn]
            v = sub["value"].values
            vn = (v - np.nanmin(v)) / (np.nanmax(v) - np.nanmin(v) + 1e-9)
            jit = (rng.random(len(sub)) - 0.5) * 0.62
            sc = ax.scatter(sub["shap"], yi + jit, c=vn, cmap=fs.DIVERGING,
                            vmin=0, vmax=1, s=4, alpha=0.55, linewidths=0)
        ax.set_yticks(range(len(feats)))
        ax.set_yticklabels([fs.pretty(f) for f in feats], fontsize=8)
        ax.axvline(0, c=fs.BASELINE, lw=0.9, zorder=1)
        ax.set_xlabel("SHAP value  (impact on the model output)", fontsize=9)
        fs.strip_spines(ax, keep=("bottom",))
        ax.tick_params(axis="y", length=0)
        fs.panel_title(ax, PANEL[i], TASK_LABEL[task])

    if sc is not None:
        cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), fraction=0.016,
                            pad=0.015, aspect=18, shrink=0.5)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["low", "high"])
        cbar.set_label("Feature value", fontsize=9, color=fs.INK_2)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(length=0, labelcolor=fs.INK_2)
    save(fig, "fig5_shap")


if __name__ == "__main__":
    fig_single_vs_ensemble("auroc", "fig2_auroc")
    fig_single_vs_ensemble("auprc", "fig3_auprc")
    fig_shap()
    print("FIGURES DONE ->", FIGURES)
