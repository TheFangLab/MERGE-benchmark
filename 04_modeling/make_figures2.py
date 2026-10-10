"""Figure 4 (per-dataset algorithm comparison) and Figure 6 (class balance).

Both describe MERGE without AlphaGenome -- the ensemble trained on the candidate
pool with every `alphagenome_*` column removed. The numbers behind Figure 6 are
Table 4 (05_results/tables/Table4_class_balance.csv).

Figures carry no titles: the caption is supplied by the manuscript. Each figure is
written as PNG and as LZW-compressed TIFF at 300 dpi.
"""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import BALANCED as BAL, FIGURES, RESULTS
import figstyle as fs

fs.apply_style()
TASKS = ["coding", "splicing", "noncoding", "nonsnv"]
LAB = {"coding": "Coding SNVs", "splicing": "Splicing SNVs",
       "noncoding": "Non-coding SNVs", "nonsnv": "Non-SNV variants"}
PANEL = ["a", "b", "c", "d"]



def save(fig, stem):
    fs.save_figure(fig, FIGURES, stem)


# ---- Figure 4 : algorithm comparison within each dataset -------------------
def fig_algorithms_per_dataset():
    """Dot-and-whisker rather than bars: within a task the ten algorithms differ by
    less than a standard deviation, and bars on a truncated baseline would make that
    difference look like a gap."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for i, task in enumerate(TASKS):
        ax = axes[i // 2][i % 2]
        d = pd.read_csv(f"{RESULTS}/{task}_algo_inner.csv")
        g = (d.groupby("algorithm")["auroc"].agg(["mean", "std"])
             .fillna({"std": 0.0}).sort_values("mean", ascending=True))

        fs.grid_x(ax)
        y = np.arange(len(g))
        for yi, (algo, row) in zip(y, g.iterrows()):
            c = fs.algo_color(algo)
            ax.plot([row["mean"] - row["std"], row["mean"] + row["std"]], [yi, yi],
                    color=c, lw=1.8, alpha=0.45, solid_capstyle="round", zorder=3)
            ax.plot(row["mean"], yi, "o", color=c, ms=8, zorder=4,
                    markeredgecolor=fs.SURFACE, markeredgewidth=1.1)
        ax.set_yticks(y)
        ax.set_yticklabels([fs.pretty_algo(a) for a in g.index], fontsize=8.5)
        ax.set_ylim(-0.7, len(g) - 0.3)
        ax.set_xlabel("Inner GroupKFold AUROC (mean ± SD)", fontsize=9)
        fs.strip_spines(ax, keep=("bottom",))
        ax.tick_params(axis="y", length=0)
        top = g.index[-1]
        ax.text(g.loc[top, "mean"] + g.loc[top, "std"], len(g) - 1,
                f"  {g.loc[top, 'mean']:.3f}", va="center", fontsize=8,
                color=fs.INK, fontweight="bold")
        fs.panel_title(ax, PANEL[i], LAB[task])

    fs.class_legend(fig, fs.ALGO_CLASS_ORDER, fs.ALGO_CLASS_COLOR, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    save(fig, "fig4_algorithms_per_dataset")


# ---- class-balance sensitivity (manuscript table) --------------------------
def _best(dirpath, task):
    m = json.load(open(f"{dirpath}/{task}_meta.json"))
    s = pd.read_csv(f"{dirpath}/{task}_merge_summary.csv")
    best = m["best_strategy"]
    r = s[s.strategy == best].iloc[0]
    return dict(auroc=r.auroc_mean, auroc_sd=r.auroc_sd,
                auprc=r.auprc_mean, auprc_sd=r.auprc_sd,
                pos_rate=m["pos_rate"], strat=best,
                single_auroc=m["best_single_auroc"],
                single_auprc=m["best_single_auprc"])


# ---- Figure 6 : what changes when the classes are balanced 1:1 -------------
def fig_class_balance():
    """Paired marks, original ratio against a 1:1 downsample.

    AUROC and AUPRC are on separate panels because they behave completely
    differently here: AUROC is a ranking statistic and barely moves, while AUPRC
    is anchored to the positive rate, so balancing a 1.5%-positive dataset to 50%
    moves it by more than half its range. The dashed mark on the AUPRC panel is
    the no-skill level for each condition, which is the whole explanation.
    """
    orig = {t: _best(RESULTS, t) for t in TASKS}
    bal = {t: _best(BAL, t) for t in TASKS}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, metric, label in zip(axes, ("auroc", "auprc"), ("AUROC", "AUPRC")):
        fs.grid_x(ax)
        y = np.arange(len(TASKS))[::-1]
        for yi, t in zip(y, TASKS):
            a, b = orig[t][metric], bal[t][metric]
            ax.plot([a, b], [yi, yi], color=fs.BASELINE, lw=2.2, zorder=2,
                    solid_capstyle="round")
            # original as a solid dot, balanced as an open ring drawn over it,
            # so the two stay visible when they coincide (as they do on AUROC)
            ax.plot(a, yi, "o", ms=8, color=fs.BLUE, zorder=4,
                    markeredgecolor=fs.SURFACE, markeredgewidth=1.1)
            ax.plot(b, yi, "o", ms=12, markerfacecolor="none", zorder=5,
                    markeredgecolor=fs.ORANGE, markeredgewidth=2.0)
            if metric == "auprc":
                ax.plot(orig[t]["pos_rate"], yi, "|", ms=11, color=fs.BLUE,
                        alpha=0.75, zorder=3)
                ax.plot(0.5, yi, "|", ms=11, color=fs.ORANGE, alpha=0.75, zorder=3)
            pad = 0.004 if metric == "auroc" else 0.03
            ax.text(max(a, b) + pad, yi, f"{b - a:+.3f}", va="center",
                    fontsize=8.2, color=fs.INK_2, fontfamily="monospace")
        ax.set_yticks(y)
        ax.set_yticklabels([LAB[t] for t in TASKS], fontsize=9)
        ax.set_ylim(-0.7, len(TASKS) - 0.3)
        if metric == "auroc":
            # every AUROC here sits above 0.9 and moves by at most 0.003; a 0-1
            # axis would draw that as a single dot, so this panel zooms in
            ax.set_xlim(0.90, 1.004)
            ax.set_xticks(np.arange(0.90, 1.001, 0.02))
        else:
            ax.set_xlim(0, 1.16)
            ax.set_xticks(np.arange(0, 1.01, 0.2))
        ax.set_xlabel(f"{label} (chromosome-held-out, mean over folds)", fontsize=9)
        fs.strip_spines(ax, keep=("bottom",))
        ax.tick_params(axis="y", length=0)
        fs.panel_title(ax, "ab"[0 if metric == "auroc" else 1], label)
    fs.class_legend(fig, ["Original class ratio", "1:1 downsample"],
                    {"Original class ratio": fs.BLUE, "1:1 downsample": fs.ORANGE},
                    bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    save(fig, "fig6_class_balance")


if __name__ == "__main__":
    fig_algorithms_per_dataset()
    if os.path.isdir(BAL):
        fig_class_balance()
    else:
        print("balanced results not found; skipped the class-balance comparison")
    print("FIGURES2 DONE")
