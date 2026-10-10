"""Shared figure style for the MERGE benchmark: palette, naming, chrome.

Colour is assigned by the job it does, not by taste:

* **categorical** — algorithm family in the algorithm figures, condition in the
  balanced comparison. Four fixed slots, assigned in a fixed order, never cycled.
* **sequential** — one hue, light to dark, for magnitude (the cross-task rank
  heat map).
* **ordinal, decorative** — the rank ramp of fig3 / fig4, where the bars are
  already sorted and colour only restates their order (see RANK_RAMP).
* **diverging** — two hues either side of a neutral grey, for polarity
  (correlation matrices, SHAP value direction).

The categorical slots were validated as a set against an all-pairs colour-vision
check on the light chart surface (`#fcfcfb`): worst deutan/protan/tritan ΔE 9.2,
worst normal-vision ΔE 16.3, both clear of the 8 / 15 floors. Do not re-pick these
by eye — if you change one, re-validate the whole set.

Aqua sits below 3:1 contrast against the surface, so every figure that uses it
carries visible labels (bar-end values or cell text) rather than relying on the
fill alone.
"""
from __future__ import annotations

import matplotlib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# --- categorical slots (validated as a set; fixed order) ----------------------
BLUE = "#2a78d6"
AQUA = "#1baf7a"
ORANGE = "#eb6834"
VIOLET = "#4a3aa7"

# --- ink and chrome -----------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# --- sequential ramp (one hue, light -> dark) ---------------------------------
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6",
             "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ_BLUE = LinearSegmentedColormap.from_list("merge_seq_blue", BLUE_RAMP)

# --- rank ramp for the single-feature figures --------------------------------
# fig3 / fig4 sort their bars by the metric, so bar length already carries the
# value; the ramp restates it and carries no information of its own. That is why a
# multi-hue ramp is acceptable here and nowhere else in this repository: nothing is
# lost if a reader cannot separate two stops. MERGE_ACCENT sits outside the ramp and
# is the one colour in those figures that does mean something.
RANK_RAMP_STOPS = ["#eda100", "#86b73f", "#1baf7a", "#1795a8", "#2a78d6", "#0d366b"]
RANK_RAMP = LinearSegmentedColormap.from_list("merge_rank", RANK_RAMP_STOPS)

# Chosen by measurement, not by eye: against every stop of the ramp above its worst
# simulated-CVD separation is dE 10.5 and its worst normal-vision separation 30.2,
# the best of the candidates tried (#e34948 bottoms out at 6.9, #d5348e at 4.7).
MERGE_ACCENT = "#c2185b"
# The optional MERGE + AlphaGenome average is drawn in the same accent hue but
# open rather than solid: it is a variant of the same model, and it is only
# available when an AlphaGenome score exists for the variant.
MERGE_AVG_FILL = "#f2ccda"
MERGE_AVG_LABEL = "MERGE + AlphaGenome average"


def rank_colors(n: int):
    """Colours for n bars sorted worst-to-best (index 0 = worst = light end)."""
    if n <= 1:
        return [RANK_RAMP(1.0)]
    return [RANK_RAMP(i / (n - 1)) for i in range(n)]


# --- diverging pair (blue <-> red, neutral grey midpoint) ---------------------
DIVERGING = LinearSegmentedColormap.from_list(
    "merge_div",
    ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f6a9a8", "#e34948", "#8f2726"],
)

# ==============================================================================
# Model classes
# ==============================================================================
CLASS_PROTEIN = "Protein & alignment-based"
CLASS_REGULATORY = "Regulatory (sequence-to-function)"
CLASS_GENOMIC_LM = "Genomic language models"
CLASS_MERGE = "MERGE ensemble"

CLASS_ORDER = [CLASS_PROTEIN, CLASS_REGULATORY, CLASS_GENOMIC_LM, CLASS_MERGE]
CLASS_COLOR = {
    CLASS_PROTEIN: BLUE,
    CLASS_REGULATORY: AQUA,
    CLASS_GENOMIC_LM: ORANGE,
    CLASS_MERGE: VIOLET,
}


def model_class(feature: str) -> str:
    """Map a feature column to the class of model that produced it."""
    f = str(feature).lower()
    if f.startswith("merge"):
        return CLASS_MERGE
    if "alphamissense" in f or "esm1b" in f or "gpn_msa" in f:
        return CLASS_PROTEIN
    if "alphagenome" in f or "enformer" in f:
        return CLASS_REGULATORY
    return CLASS_GENOMIC_LM


def feature_color(feature: str) -> str:
    return CLASS_COLOR[model_class(feature)]


# ==============================================================================
# Display names
# ==============================================================================
_EXACT = {
    "AlphaMissense_score": "AlphaMissense",
    "ESM1b_score": "ESM-1b",
    "GPN_MSA_score": "GPN-MSA",
    "evo2_7b_score": "Evo2 7B",
    "evo2_7b_base_score": "Evo2 7B base",
    "evo2_1b_base_score": "Evo2 1B base",
    "carbon_score": "Carbon-3B",
    "generator_score": "GENERator",
    "generator_v2_score": "GENERator v2",
    "NT_score": "NT 500M",
    "nt_old_score": "NT 500M",
    "nt_v2_score": "NT v2 500M",
    "heyenadna_score": "HyenaDNA",
    "hyenadna_score": "HyenaDNA",
    "alphagenome_splicing": "AlphaGenome splicing",
    "enformer_absmean": "Enformer abs-mean",
    "enformer_mean": "Enformer mean",
    "enformer_std": "Enformer SD",
    "enformer_max_abs": "Enformer max |Δ|",
    "enformer_top5_absmean": "Enformer top-5 abs-mean",
    "enformer_pos_sum": "Enformer positive sum",
    "enformer_neg_sum": "Enformer negative sum",
}

_ALGO = {
    "LR_L2": "Logistic (L2)",
    "LR_L1": "Logistic (L1)",
    "ElasticNet": "Elastic net",
    "RandomForest": "Random forest",
    "ExtraTrees": "Extra trees",
    "HistGB": "HistGradientBoosting",
}

_STRATEGY = {"beam": "beam search", "lasso": "LASSO", "correlation": "correlation filter"}


def pretty(feature: str) -> str:
    """Human-readable name for a feature column."""
    f = str(feature)
    if f in _EXACT:
        return _EXACT[f]
    if f.startswith("alphagenome_"):
        # alphagenome_{raw,quantile}_score_{max,min,mean}
        parts = f.split("_")
        if len(parts) >= 4:
            return f"AlphaGenome {parts[1]} {parts[-1]}"
    return f.replace("_score", "").replace("_", " ")


def pretty_algo(name: str) -> str:
    return _ALGO.get(str(name), str(name))


def pretty_strategy(name: str) -> str:
    return _STRATEGY.get(str(name), str(name))


def merge_label(strategy: str) -> str:
    return f"MERGE ({pretty_strategy(strategy)})"


# ==============================================================================
# Algorithm families (categorical, same four slots)
# ==============================================================================
ALGO_LINEAR = "Linear"
ALGO_BAGGING = "Bagged trees"
ALGO_BOOSTING = "Gradient boosting"
ALGO_NEURAL = "Neural"

ALGO_CLASS_ORDER = [ALGO_LINEAR, ALGO_BAGGING, ALGO_BOOSTING, ALGO_NEURAL]
ALGO_CLASS_COLOR = {
    ALGO_LINEAR: BLUE,
    ALGO_BAGGING: AQUA,
    ALGO_BOOSTING: ORANGE,
    ALGO_NEURAL: VIOLET,
}


def algo_class(name: str) -> str:
    n = str(name).lower()
    if n.startswith("lr_") or "elasticnet" in n:
        return ALGO_LINEAR
    if "randomforest" in n or "extratrees" in n:
        return ALGO_BAGGING
    if "mlp" in n:
        return ALGO_NEURAL
    return ALGO_BOOSTING


def algo_color(name: str) -> str:
    return ALGO_CLASS_COLOR[algo_class(name)]


# ==============================================================================
# Chrome
# ==============================================================================
def apply_style() -> None:
    """Global rcParams. Call once, before creating any figure."""
    matplotlib.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "axes.titlecolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "legend.frameon": False,
        "legend.handlelength": 1.1,
        "legend.handleheight": 1.1,
        "legend.borderpad": 0.2,
        "savefig.bbox": "tight",
        "figure.dpi": 110,
    })


def panel_title(ax, letter: str, text: str, pad: float = 8) -> None:
    """Bold panel letter followed by a regular-weight description."""
    ax.set_title(f"{letter}  {text}", loc="left", fontsize=11,
                 fontweight="bold", color=INK, pad=pad)


def grid_x(ax) -> None:
    ax.grid(axis="x", color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)


def grid_y(ax) -> None:
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)


def strip_spines(ax, keep=("left", "bottom")) -> None:
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def class_legend(fig, classes, colors, **kw):
    """One legend for the whole figure, so identity is never colour-alone."""
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=colors[c], edgecolor="none", label=c) for c in classes]
    defaults = dict(loc="lower center", ncol=len(classes), fontsize=9,
                    bbox_to_anchor=(0.5, -0.01), frameon=False)
    defaults.update(kw)
    return fig.legend(handles=handles, **defaults)


# ==============================================================================
# Rounded bar ends
# ==============================================================================
def round_bar_ends(ax, bars, orientation: str = "h", frac: float = 0.45) -> None:
    """Replace rectangular bars with bars rounded at the data end only.

    The rounding radius is computed through the axes transform so it renders as a
    true semicircle regardless of the figure's aspect ratio. Call after the axis
    limits are final.
    """
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    ax.figure.canvas.draw()
    origin = ax.transData.transform((0.0, 0.0))
    px_per_x = abs(ax.transData.transform((1.0, 0.0))[0] - origin[0]) or 1.0
    px_per_y = abs(ax.transData.transform((0.0, 1.0))[1] - origin[1]) or 1.0
    k = 0.5523  # circular-arc Bezier constant

    for bar in list(bars):
        x0, y0 = bar.get_x(), bar.get_y()
        w, h = bar.get_width(), bar.get_height()
        color = bar.get_facecolor()
        edge = bar.get_edgecolor()
        lw = bar.get_linewidth()
        if w == 0 or h == 0:
            continue

        if orientation == "h":
            r_px = min(abs(h) * px_per_y * frac, abs(w) * px_per_x * 0.5)
            rx, ry = r_px / px_per_x, r_px / px_per_y
            sign = 1.0 if w >= 0 else -1.0
            rx *= sign
            xe, yb, yt = x0 + w, y0, y0 + h
            verts = [
                (x0, yb), (xe - rx, yb),
                (xe - rx + k * rx, yb), (xe, yb + ry - k * ry), (xe, yb + ry),
                (xe, yt - ry),
                (xe, yt - ry + k * ry), (xe - rx + k * rx, yt), (xe - rx, yt),
                (x0, yt), (x0, yb),
            ]
        else:
            r_px = min(abs(w) * px_per_x * frac, abs(h) * px_per_y * 0.5)
            rx, ry = r_px / px_per_x, r_px / px_per_y
            sign = 1.0 if h >= 0 else -1.0
            ry *= sign
            xl, xr, yb, ye = x0, x0 + w, y0, y0 + h
            verts = [
                (xl, yb), (xl, ye - ry),
                (xl, ye - ry + k * ry), (xl + rx - k * rx, ye), (xl + rx, ye),
                (xr - rx, ye),
                (xr - rx + k * rx, ye), (xr, ye - ry + k * ry), (xr, ye - ry),
                (xr, yb), (xl, yb),
            ]

        codes = [Path.MOVETO, Path.LINETO,
                 Path.CURVE4, Path.CURVE4, Path.CURVE4,
                 Path.LINETO,
                 Path.CURVE4, Path.CURVE4, Path.CURVE4,
                 Path.LINETO, Path.CLOSEPOLY]
        ax.add_patch(PathPatch(Path(verts, codes), facecolor=color,
                               edgecolor=edge, linewidth=lw,
                               zorder=bar.get_zorder()))
        bar.set_visible(False)


def save_figure(fig, outdir: str, stem: str, png_dpi: int = 200, tiff_dpi: int = 300) -> None:
    """Write <stem>.png (screen) and <stem>.tiff (submission: LZW, 300 dpi, no alpha)."""
    import os
    import matplotlib.pyplot as plt
    from PIL import Image
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, f"{stem}.png"), dpi=png_dpi, bbox_inches="tight")
    tif = os.path.join(outdir, f"{stem}.tiff")
    fig.savefig(tif, dpi=tiff_dpi, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    with Image.open(tif) as im:                 # journals reject alpha channels
        if im.mode == "RGBA":
            flat = Image.new("RGB", im.size, SURFACE)
            flat.paste(im, mask=im.split()[3])
            flat.save(tif, compression="tiff_lzw", dpi=(tiff_dpi, tiff_dpi))
    print("wrote", os.path.join(outdir, stem) + ".{png,tiff}")
