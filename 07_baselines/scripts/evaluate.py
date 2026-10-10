"""Compare MERGE against the external baselines, per dataset.

Contract for the extractors: each writes a tab-separated file

    07_baselines/scores/<task>__<tool>.tsv     chr  pos  ref  alt  score

with one row per unique variant key and a "score" column oriented so that a
larger value means more likely pathogenic. Missing variants are simply absent
from the file; they are reported as reduced coverage, never imputed. The one
exception is AlphaGenome, whose score files carry the benchmark's own
out-of-fold percentile, with missing scores already set to the training-fold
median (see extract_alphagenome_feature.py).

Two MERGE columns are reported side by side:

    MERGE                        05_results/<task>_merge_oof.csv
    MERGE + AlphaGenome average  05_results_alphagenome_average/
                                 <task>_alphagenome_average_oof.csv

MERGE is trained on the candidate pool with every `alphagenome_*` column
removed, so AlphaGenome is an ordinary external comparator here rather than one
of the ensemble's own ingredients. The average is offered on splicing and
non-coding variants only; its AlphaGenome half uses the same training-fold
median imputation, so it scores every variant.

Both are out-of-fold predictions from the chromosome-held-out 3-fold CV, so no
baseline is being compared against a model that saw the same variants.

Every metric is computed per fold and averaged, matching the convention of the
existing results tables, and every row is computed on the *common* subset --
the variants that every predictor in that table can score -- as well as on the
tool's own covered subset, so that a tool is never credited for a subset that
happens to be easier.

Usage: python evaluate.py [repo-root]
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

# Only consult argv when this file is the program being run. Importing it from
# another script must not reinterpret that script's arguments as a repo path.
_DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
ROOT = os.path.abspath(
    sys.argv[1] if (__name__ == "__main__" and len(sys.argv) > 1) else _DEFAULT_ROOT)
SCORES = os.path.join(ROOT, "07_baselines", "scores")
OUT = os.path.join(ROOT, "07_baselines", "results")

DATASETS = {"coding": "coding_features.csv", "splicing": "splicing_features.csv",
            "noncoding": "noncoding_features.csv", "nonsnv": "non_snv_features.csv"}

MERGE_RUNS = {"MERGE": "05_results",
              "MERGE + AlphaGenome average": "05_results_alphagenome_average"}

# Chromosome-held-out folds, identical to 04_modeling/config.py.
FOLDS = {1: {"3", "6", "9", "12", "16", "22", "X", "Y"},
         2: {"2", "5", "11", "14", "17", "18", "19", "20", "21"},
         3: {"1", "4", "7", "8", "10", "13", "15"}}

# Tier A -- not trained on clinical pathogenicity labels, so a fair head-to-head
# with MERGE, which uses only label-free features.
# Tier B -- trained on ClinVar and/or HGMD labels, so partly circular against a
# ClinVar-derived benchmark. Reported, but never used to claim MERGE "wins".
TIER = {
    "CADD": "A", "phyloP100way": "A", "phyloP470way": "A", "GERP++": "A",
    "LINSIGHT": "A", "SpliceAI": "A", "Pangolin": "A", "PrimateAI": "A",
    "SIFT4G": "A", "Polyphen2_HVAR": "A", "PROVEAN": "A",
    "MutationAssessor": "A", "frameshift_rule": "A",
    # dbNSFP copies of scores also obtained directly; kept as a cross-check that
    # the two extraction routes agree.
    "CADD_dbnsfp": "A", "phyloP100way_dbnsfp": "A", "phyloP470way_dbnsfp": "A",
    "ReMM": "B", "ReMM_v031": "B", "ReMM_v04": "B", "ncER": "B", "NCBoost": "B", "GWAVA": "B", "REVEL": "B",
    "ClinPred": "B", "MetaRNN": "B", "BayesDel": "B", "M-CAP": "B",
    "MutFormer": "B",
    "VEST4": "B", "VARITY": "B", "dbscSNV_ADA": "B", "dbscSNV_RF": "B",
    "SQUIRLS": "B", "CAPICE": "B",
    # AlphaGenome is now an ordinary external comparator: MERGE is trained
    # without it, so it is no longer "the ensemble's own best ingredient". It is
    # trained on functional-genomics measurements rather than clinical labels,
    # which puts it in tier A. Its score is the raw AlphaGenome column mapped to
    # a percentile -- a rank transform, so AUROC and AUPRC are the raw column's.
    "AlphaGenome": "A",
    "AlphaGenome_raw_max": "A", "AlphaGenome_quantile_max": "A",
}


def load_dataset(task):
    df = pd.read_csv(os.path.join(ROOT, "03_datasets", DATASETS[task]),
                     usecols=["chr", "pos", "ref", "alt", "label"], low_memory=False)
    df["chr"] = df["chr"].astype(str).str.replace("chr", "", regex=False)
    fold = pd.Series(-1, index=df.index)
    for k, chroms in FOLDS.items():
        fold[df["chr"].isin(chroms)] = k
    df["fold"] = fold
    return df


def merge_oof(task, run_dir):
    """Out-of-fold MERGE probabilities for the task's best strategy.

    Aligned by row position, never by key. All four datasets carry unique
    (chr, pos, ref, alt) keys since the strict deduplication of the splicing
    table (35,774 -> 35,015), but position alignment is what the OOF files
    actually guarantee, so it stays the contract.
    """
    avg_p = os.path.join(ROOT, run_dir, f"{task}_alphagenome_average_oof.csv")
    if os.path.exists(avg_p):                       # the optional average
        d = pd.read_csv(avg_p, low_memory=False)
        return d["merge_alphagenome_average"].to_numpy(), "1:1 average"
    meta_p = os.path.join(ROOT, run_dir, f"{task}_meta.json")
    oof_p = os.path.join(ROOT, run_dir, f"{task}_merge_oof.csv")
    if not (os.path.exists(meta_p) and os.path.exists(oof_p)):
        return None, None
    meta = json.load(open(meta_p))
    oof = pd.read_csv(oof_p, low_memory=False)
    strat = meta["best_strategy"]
    return oof[strat].to_numpy(), strat


def load_scores(task):
    """Read every extractor output for the task, mapped onto dataset rows."""
    out = {}
    if not os.path.isdir(SCORES):
        return out
    for fn in sorted(os.listdir(SCORES)):
        if not fn.startswith(f"{task}__") or not fn.endswith(".tsv"):
            continue
        tool = fn[len(task) + 2:-4]
        s = pd.read_csv(os.path.join(SCORES, fn), sep="\t", low_memory=False)
        s["chr"] = s["chr"].astype(str).str.replace("chr", "", regex=False)
        key = list(zip(s["chr"], s["pos"], s["ref"], s["alt"]))
        out[tool] = dict(zip(key, pd.to_numeric(s["score"], errors="coerce")))
    return out


def per_fold(y, p, folds, mask):
    """Mean and sd of AUROC / AUPRC over the folds, restricted to `mask`."""
    aurocs, auprcs = [], []
    for k in sorted(FOLDS):
        sel = mask & (folds == k) & ~np.isnan(p)
        if sel.sum() < 10 or len(np.unique(y[sel])) < 2:
            continue
        aurocs.append(roc_auc_score(y[sel], p[sel]))
        auprcs.append(average_precision_score(y[sel], p[sel]))
    if not aurocs:
        return dict(auroc=np.nan, auroc_sd=np.nan, auprc=np.nan, auprc_sd=np.nan)
    # sample SD over the outer folds (divisor k - 1)
    return dict(auroc=float(np.mean(aurocs)), auroc_sd=float(np.std(aurocs, ddof=1)),
                auprc=float(np.mean(auprcs)), auprc_sd=float(np.std(auprcs, ddof=1)))


def run_task(task):
    df = load_dataset(task)
    df = df[df["fold"] != -1].reset_index(drop=True)
    y = df["label"].to_numpy().astype(int)
    folds = df["fold"].to_numpy()
    keys = list(zip(df["chr"], df["pos"], df["ref"], df["alt"]))

    preds, strategies = {}, {}
    for label, run_dir in MERGE_RUNS.items():
        p, strat = merge_oof(task, run_dir)
        if p is None:
            print(f"  [{task}] {label}: not available yet, skipped")
            continue
        if len(p) != len(df):
            raise SystemExit(f"{task}: {label} OOF has {len(p)} rows, "
                             f"dataset has {len(df)} -- row alignment broken")
        preds[label] = p
        strategies[label] = strat

    for tool, lookup in load_scores(task).items():
        preds[tool] = np.array([lookup.get(k, np.nan) for k in keys], dtype=float)

    if not preds:
        return None

    # The common subset: variants every predictor in this table can score.
    common = np.ones(len(df), dtype=bool)
    for p in preds.values():
        common &= ~np.isnan(p)

    rows = []
    for name, p in preds.items():
        own = ~np.isnan(p)
        r = {"task": task, "predictor": name,
             "tier": "MERGE" if name.startswith("MERGE") else TIER.get(name, "?"),
             "strategy": strategies.get(name, ""),
             "n_scored": int(own.sum()), "coverage": round(own.mean(), 4),
             "n_common": int(common.sum()), "positives_common": int(y[common].sum())}
        for suffix, mask in (("_common", common), ("_own", own)):
            for k, v in per_fold(y, p, folds, mask).items():
                r[k + suffix] = None if np.isnan(v) else round(v, 4)
        rows.append(r)

    res = pd.DataFrame(rows)
    res["_o"] = res["predictor"].str.startswith("MERGE").map({True: 0, False: 1})
    res = res.sort_values(["_o", "auroc_common"], ascending=[True, False]).drop(columns="_o")
    print(f"\n=== {task}  n={len(df)}  positives={y.sum()} "
          f"common subset={int(common.sum())} ===")
    print(res[["predictor", "tier", "n_scored", "coverage",
               "auroc_common", "auprc_common", "auroc_own", "auprc_own"]]
          .to_string(index=False))
    return res


# Region classes large enough to carry a stable estimate. UTR3 (18 positives),
# downstream (0) and intergenic (2) are counted in the composition table but not
# scored -- an AUPRC on 18 positives says nothing.
STRATA_MIN_POSITIVES = 40


def run_strata(task="noncoding"):
    """Per-region-class comparison, so a pooled score cannot hide composition.

    A non-coding set pooled across region classes inflates AUROC for every
    method when the region composition differs between positives and negatives
    (Grosser et al., bioRxiv 2025.09.05.674459). Here it mostly does not -- the
    positive rate is 1.7 % intronic, 1.6 % UTR5, 2.5 % upstream against 1.5 %
    overall -- with one exception: UTR3 is 10 % of the data at a 0.2 % positive
    rate. Reporting the strata makes that visible instead of arguable.
    """
    region_file = os.path.join(SCORES, f"{task}_region.tsv")
    if not os.path.exists(region_file):
        return None
    reg = pd.read_csv(region_file, sep="\t", low_memory=False)
    reg["chr"] = reg["chr"].astype(str).str.replace("chr", "", regex=False)
    lookup = dict(zip(zip(reg["chr"], reg["pos"], reg["ref"], reg["alt"]),
                      reg["region"]))

    df = load_dataset(task)
    df = df[df["fold"] != -1].reset_index(drop=True)
    y = df["label"].to_numpy().astype(int)
    folds = df["fold"].to_numpy()
    keys = list(zip(df["chr"], df["pos"], df["ref"], df["alt"]))
    regions = np.array([lookup.get(k, "unknown") for k in keys])

    preds = {}
    for label, run_dir in MERGE_RUNS.items():
        p, _ = merge_oof(task, run_dir)
        if p is not None and len(p) == len(df):
            preds[label] = p
    preds.update({t: np.array([lk.get(k, np.nan) for k in keys], dtype=float)
                  for t, lk in load_scores(task).items()})
    if not preds:
        return None

    rows = []
    for region in sorted(set(regions)):
        sel = regions == region
        if y[sel].sum() < STRATA_MIN_POSITIVES:
            continue
        for name, p in preds.items():
            m = per_fold(y, p, folds, sel & ~np.isnan(p))
            rows.append({"task": task, "region": region, "predictor": name,
                         "n": int(sel.sum()), "positives": int(y[sel].sum()),
                         "auroc": None if np.isnan(m["auroc"]) else round(m["auroc"], 4),
                         "auprc": None if np.isnan(m["auprc"]) else round(m["auprc"], 4)})
    res = pd.DataFrame(rows)
    print(f"\n=== {task}, by region class ===")
    print(res.pivot_table(index=["region", "n", "positives"],
                          columns="predictor", values="auprc")
          .to_string())
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    allres = [r for r in (run_task(t) for t in DATASETS) if r is not None]
    if allres:
        out = pd.concat(allres, ignore_index=True)
        out.to_csv(os.path.join(OUT, "baseline_comparison.csv"), index=False)
        print(f"\nwrote {os.path.join(OUT, 'baseline_comparison.csv')}")
    strata = run_strata("noncoding")
    if strata is not None:
        p = os.path.join(OUT, "baseline_comparison_by_region.csv")
        strata.to_csv(p, index=False)
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
