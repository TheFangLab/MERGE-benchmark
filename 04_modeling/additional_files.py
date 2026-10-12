"""Write Additional files 1-3 of the manuscript (Tables S1-S23) as .xlsx workbooks.

    python additional_files.py OUTDIR

Every number is read from the committed results (05_results/, 05_results_alphagenome_average/,
07_baselines/results/, 08_external_validation/) or recomputed from the committed
out-of-fold predictions; the per-fold single-feature metrics are checked against
{task}_singlefeature.csv before anything is written. Requires openpyxl.

Additional file 1  Individual model benchmarks                  Tables S1-S2
Additional file 2  MERGE integration and assessment             Tables S3-S15
Additional file 3  External comparisons and BRCA1 validation    Tables S16-S23
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font
from sklearn.metrics import average_precision_score, roc_auc_score

from config import AGAVG, BALANCED, DATA, FOLDS, REPO, RESULTS, TASKS, ENSEMBLE_EXCLUDE_PREFIXES
from figstyle import pretty as _figure_label, pretty_algo
from manuscript_tables import NAME as _TABLE_NAME

_NAME = {**_TABLE_NAME, "generator_score": "GENERATOR", "generator_v2_score": "GENERATOR-v2"}


def pretty(col):
    """Feature names as in the manuscript text and tables; figure labels otherwise."""
    return _NAME.get(col, _figure_label(col))

def read_csv(path, **kw):
    """Exact float parsing: the default C parser can change the last bit of a value, which
    breaks near-ties in saturated scores and moves AUPRC by up to 1e-6."""
    return pd.read_csv(path, float_precision="round_trip", low_memory=False, **{k: v for k, v in kw.items() if k != "low_memory"})


TASK_NAME = {"coding": "Coding SNVs", "splicing": "Splicing SNVs",
             "noncoding": "Noncoding SNVs", "nonsnv": "Non-SNVs"}
STRATEGY = {"beam": "beam search", "lasso": "LASSO", "correlation": "correlation filtering"}
BASE = os.path.join(str(REPO), "07_baselines", "results")
SGE = os.path.join(str(REPO), "08_external_validation")


def chrom_key(c):
    c = str(c).replace("chr", "")
    return (0, int(c)) if c.isdigit() else (1, c)


FOLD_CHROMS = {k: ", ".join(sorted(v, key=chrom_key)) for k, v in FOLDS.items()}


def fold_of(chrom):
    c = str(chrom).replace("chr", "")
    return next((k for k, v in FOLDS.items() if c in v), -1)


def is_ag(col):
    return any(col.startswith(p) for p in ENSEMBLE_EXCLUDE_PREFIXES)


def meta(task, balanced=False):
    return json.load(open(os.path.join(BALANCED if balanced else RESULTS, f"{task}_meta.json")))


def r4(df, cols=None, nd=4):
    cols = cols or [c for c in df.columns if df[c].dtype.kind == "f"]
    for c in cols:
        df[c] = df[c].round(nd)
    return df


# ───────────────────────── Additional file 1 ─────────────────────────

def file1():
    s1, s2 = [], []
    for task, fname in TASKS.items():
        oof = read_csv(os.path.join(RESULTS, f"{task}_singlefeature_oof.csv"), low_memory=False)
        data = read_csv(os.path.join(DATA, fname), low_memory=False)
        data["fold"] = data["chr"].map(fold_of)
        data = data[data["fold"] != -1].reset_index(drop=True)
        assert len(data) == len(oof) and (data["pos"].values == oof["pos"].values).all()
        summ = read_csv(os.path.join(RESULTS, f"{task}_singlefeature.csv")).set_index("feature")
        y, fold = oof["label"].values, data["fold"].values
        for feat in summ.index:
            p = oof[feat].values
            per = [(k, roc_auc_score(y[fold == k], p[fold == k]),
                    average_precision_score(y[fold == k], p[fold == k])) for k in sorted(FOLDS)]
            au = np.array([a for _, a, _ in per]); ap = np.array([b for _, _, b in per])
            # the per-fold numbers must reproduce the committed summary exactly
            assert abs(au.mean() - summ.loc[feat, "auroc_mean"]) < 1e-12, (task, feat)
            assert abs(ap.mean() - summ.loc[feat, "auprc_mean"]) < 1e-12, (task, feat)
            obs = int(data[feat].notna().sum())
            s1.append({"Task": TASK_NAME[task], "Feature": pretty(feat), "Column": feat,
                       "AlphaGenome-derived": "yes" if is_ag(feat) else "no",
                       "MERGE candidate": "no" if is_ag(feat) else "yes",
                       "Held-out scoring": ("label-free percentile against the training folds"
                                            if summ.loc[feat, "scored_by"] == "rank"
                                            else "logistic regression fitted on the training folds"),
                       "Variants": len(data), "Positives": int(y.sum()),
                       "Observed scores": obs, "Missing (%)": round(100 * (1 - obs / len(data)), 2),
                       "AUROC, fold mean": au.mean(), "AUROC, sample SD": au.std(ddof=1),
                       "AUPRC, fold mean": ap.mean(), "AUPRC, sample SD": ap.std(ddof=1),
                       "AUROC, pooled": roc_auc_score(y, p), "AUPRC, pooled": average_precision_score(y, p)})
            for k, a, b in per:
                s2.append({"Task": TASK_NAME[task], "Feature": pretty(feat), "Column": feat, "Fold": k,
                           "Held-out chromosomes": FOLD_CHROMS[k], "Variants": int((fold == k).sum()),
                           "Positives": int(y[fold == k].sum()), "AUROC": a, "AUPRC": b})
    s1 = pd.DataFrame(s1)
    s1["Rank in task (fold-mean AUROC)"] = s1.groupby("Task")["AUROC, fold mean"].rank(ascending=False, method="min").astype(int)
    s1 = s1.sort_values(["Task", "Rank in task (fold-mean AUROC)"], key=lambda c: c.map(
        {v: i for i, v in enumerate(TASK_NAME.values())}) if c.name == "Task" else c)
    return [("Table S1", "Individual-feature performance: fold means, sample SDs and pooled out-of-fold metrics "
             "for all 25 coding, 18 splicing, 23 noncoding and 22 non-SNV features", r4(s1)),
            ("Table S2", "Fold-specific AUROC and AUPRC of every individual feature", r4(pd.DataFrame(s2)))]


# ───────────────────────── Additional file 2 ─────────────────────────

def file2():
    out = []
    # S3 strategies
    rows = []
    for task in TASKS:
        m = meta(task); s = read_csv(os.path.join(RESULTS, f"{task}_merge_summary.csv"))
        for _, r in s.iterrows():
            rows.append({"Task": TASK_NAME[task], "Feature selection": STRATEGY[r.strategy],
                         "Task-level choice": "yes" if r.strategy == m["best_strategy"] else "no",
                         "AUROC, fold mean": r.auroc_mean, "AUROC, sample SD": r.auroc_sd,
                         "AUPRC, fold mean": r.auprc_mean, "AUPRC, sample SD": r.auprc_sd,
                         "Accuracy": r.acc, "Balanced accuracy": r.bacc, "F1": r.f1, "Precision": r.precision,
                         "Recall": r.recall, "Specificity": r.specificity, "MCC": r.mcc})
    out.append(("Table S3", "MERGE under the three feature-selection strategies (fold means; threshold metrics at 0.5). "
                "The task-level MERGE is the strategy with the highest mean outer-fold AUROC", r4(pd.DataFrame(rows))))
    # S4 average
    a = read_csv(os.path.join(AGAVG, "alphagenome_average_summary.csv"))
    rows = []
    for _, r in a.iterrows():
        miss = ""
        if r.offered == "yes":
            o = read_csv(os.path.join(AGAVG, f"{r.task}_alphagenome_average_oof.csv"), usecols=["alphagenome_observed"])
            miss = int((o.alphagenome_observed == 0).sum())
        rows.append({"Task": TASK_NAME[r.task], "Offered": r.offered, "Score": r.predictor,
                     "AlphaGenome column": r.selected_alphagenome_feature if r.offered == "yes" else "",
                     "Variants": r.n, "Positive rate": r.positive_rate,
                     "AlphaGenome scores missing (training-fold median)": miss,
                     "AUROC, fold mean": r.auroc, "AUROC, sample SD": r.auroc_sd,
                     "AUPRC, fold mean": r.auprc, "AUPRC, sample SD": r.auprc_sd,
                     "Note": r.evaluation if isinstance(r.evaluation, str) else ""})
    out.append(("Table S4", "Fixed MERGE + AlphaGenome average (0.5 x MERGE + 0.5 x AlphaGenome percentile), MERGE and "
                "the AlphaGenome component on the same complete cohorts", r4(pd.DataFrame(rows))))
    # S5 cut-points
    th = json.load(open(os.path.join(AGAVG, "merge_final_thresholds.json")))
    rows = []
    for task in ("splicing", "noncoding"):
        t = th[task]
        for band, c in zip(t["band_names"], t["per_cut_check"]):
            rows.append({"Task": TASK_NAME[task], "Band at or above the cut": band,
                         "MERGE cut": c["ensemble_cut"], "Average cut": c["merge_alphagenome_average_cut"],
                         "Fraction of variants below the cut": c["share_below_cut"],
                         "Call rate": c["ensemble"]["call_rate"],
                         "MERGE sensitivity": c["ensemble"]["sensitivity"], "MERGE specificity": c["ensemble"]["specificity"],
                         "Average sensitivity": c["merge_alphagenome_average"]["sensitivity"],
                         "Average specificity": c["merge_alphagenome_average"]["specificity"],
                         "Variants used (AlphaGenome observed)": t["n_oof"],
                         "Band agreement with MERGE": t["band_agreement_with_ensemble"],
                         "Band agreement if MERGE cuts were reused": t["band_agreement_if_reusing_ensemble_cuts"]})
    out.append(("Table S5", "Interpretation cut-points of the MERGE + AlphaGenome average, matched to the MERGE bands by "
                "call rate on out-of-fold predictions (labels are used only to report sensitivity and specificity)",
                r4(pd.DataFrame(rows))))
    # S6 outer folds
    rows = []
    for task in TASKS:
        m = meta(task); f = read_csv(os.path.join(RESULTS, f"{task}_merge_folds.csv"))
        for _, r in f.iterrows():
            rows.append({"Task": TASK_NAME[task], "Feature selection": STRATEGY[r.strategy],
                         "Task-level choice": "yes" if r.strategy == m["best_strategy"] else "no",
                         "Fold": int(r.fold), "Held-out chromosomes": FOLD_CHROMS[int(r.fold)],
                         "Algorithm": pretty_algo(r.algorithm), "Features selected": int(r.n_selected),
                         "AUROC": r.auroc, "AUPRC": r.auprc, "Accuracy": r.accuracy,
                         "Balanced accuracy": r.balanced_acc, "F1": r.f1, "Precision": r.precision,
                         "Recall": r.recall, "Specificity": r.specificity, "MCC": r.mcc})
    out.append(("Table S6", "Outer-fold MERGE results for every feature-selection strategy", r4(pd.DataFrame(rows))))
    # S7 class balance
    rows = []
    for task in TASKS:
        for bal in (False, True):
            m = meta(task, bal); base = BALANCED if bal else RESULTS
            s = read_csv(os.path.join(base, f"{task}_merge_summary.csv")).set_index("strategy")
            f = read_csv(os.path.join(base, f"{task}_merge_folds.csv"))
            for st in ("lasso", "correlation", "beam"):
                common = {"Task": TASK_NAME[task], "Class ratio": "1:1 (majority downsampled)" if bal else "original",
                          "Variants": m["n"], "Positive rate": m["pos_rate"], "No-skill AUPRC": m["pos_rate"],
                          "Feature selection": STRATEGY[st], "Task-level choice": "yes" if st == m["best_strategy"] else "no"}
                rows.append({**common, "Fold": "mean (sample SD)", "Algorithm": "", "Features selected": "",
                             "AUROC": s.loc[st, "auroc_mean"], "AUROC SD": s.loc[st, "auroc_sd"],
                             "AUPRC": s.loc[st, "auprc_mean"], "AUPRC SD": s.loc[st, "auprc_sd"]})
                for _, r in f[f.strategy == st].iterrows():
                    rows.append({**common, "Fold": int(r.fold), "Algorithm": pretty_algo(r.algorithm),
                                 "Features selected": int(r.n_selected), "AUROC": r.auroc, "AUROC SD": np.nan,
                                 "AUPRC": r.auprc, "AUPRC SD": np.nan})
    out.append(("Table S7", "Class-balance sensitivity: the full pipeline at the original class ratio and after "
                "majority-class downsampling to 1:1", r4(pd.DataFrame(rows))))
    # S8 panels
    rows = []
    for task in TASKS:
        m = meta(task); f = read_csv(os.path.join(RESULTS, f"{task}_merge_folds.csv"))
        for st in ("lasso", "correlation", "beam"):
            for k, feats in m["selected_features"][st].items():
                algo = f[(f.strategy == st) & (f.fold == int(k))].algorithm.iloc[0]
                rows.append({"Task": TASK_NAME[task], "Feature selection": STRATEGY[st],
                             "Task-level choice": "yes" if st == m["best_strategy"] else "no",
                             "Fold": int(k), "Algorithm": pretty_algo(algo), "Features selected": len(feats),
                             "Features": "; ".join(pretty(x) for x in feats), "Columns": "; ".join(feats)})
        import joblib
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # reading the panel only
            b = joblib.load(os.path.join(RESULTS, "trained_models", f"MERGE_noAlphaGenome_{task}.pkl"))
        rows.append({"Task": TASK_NAME[task], "Feature selection": STRATEGY[b["feature_selection"]],
                     "Task-level choice": "deployed model", "Fold": "all variants",
                     "Algorithm": pretty_algo(b["algorithm"]), "Features selected": len(b["features"]),
                     "Features": "; ".join(pretty(x) for x in b["features"]), "Columns": "; ".join(b["features"])})
    out.append(("Table S8", "Selected feature panels and algorithms per outer fold and strategy, and the deployed models "
                "refitted on all variants. No panel contains an AlphaGenome-derived feature", pd.DataFrame(rows)))
    # S9 inner CV
    rows = []
    for task in TASKS:
        a = read_csv(os.path.join(RESULTS, f"{task}_algo_inner.csv")).pivot(index="algorithm", columns="fold", values="auroc")
        a["mean"] = a.mean(axis=1); a["sd"] = a[[1, 2, 3]].std(axis=1, ddof=1)
        a = a.sort_values("mean", ascending=False)
        for rank, (alg, r) in enumerate(a.iterrows(), 1):
            rows.append({"Task": TASK_NAME[task], "Algorithm": pretty_algo(alg), "Fold 1": r[1], "Fold 2": r[2],
                         "Fold 3": r[3], "Mean": r["mean"], "Sample SD": r["sd"], "Rank": rank})
    out.append(("Table S9", "Inner-cross-validation AUROC of the ten candidate algorithms on each outer fold's LASSO panel "
                "(Fig. 4)", r4(pd.DataFrame(rows))))
    # S10 SHAP
    rows = []
    for task in TASKS:
        sh = read_csv(os.path.join(RESULTS, f"{task}_shap.csv")).sort_values("mean_abs_shap", ascending=False)
        for rank, (_, r) in enumerate(sh.iterrows(), 1):
            rows.append({"Task": TASK_NAME[task], "Feature selection": STRATEGY[meta(task)["best_strategy"]],
                         "Feature": pretty(r.feature), "Column": r.feature, "Mean |SHAP|": r.mean_abs_shap, "Rank": rank})
    out.append(("Table S10", "Mean absolute SHAP values of the task-level panels (LightGBM surrogate fitted within each "
                "training fold, aggregated over the held-out folds; Fig. 5)", r4(pd.DataFrame(rows))))
    # S11 calibration
    rows = []
    for c in json.load(open(os.path.join(RESULTS, "calibration", "calibration.json"))):
        for b in c["reliability"]:
            rows.append({"Task": TASK_NAME[c["task"]], "Feature selection": STRATEGY[c["strategy"]],
                         "Variants": c["n"], "Positive prevalence": c["base_rate"], "Brier score": c["brier"],
                         "Brier, base-rate predictor": c["brier_baseline"], "ECE": c["ece"], "MCE": c["mce"],
                         "Score bin": f"{b[0] - 0.05:.1f}-{b[0] + 0.05:.1f}", "Mean predicted score": b[1],
                         "Observed positive fraction": b[2], "Variants in bin": b[3]})
    out.append(("Table S11", "Calibration of the pooled out-of-fold MERGE scores: Brier score, expected (ECE) and maximum "
                "(MCE) calibration error over ten equal-width bins, and the reliability bins", r4(pd.DataFrame(rows))))
    # S12 operating points
    rows = []
    for task in TASKS:
        m = meta(task); f = read_csv(os.path.join(RESULTS, f"{task}_merge_folds.csv"))
        f = f[f.strategy == m["best_strategy"]]
        cols = ["balanced_acc", "f1", "precision", "recall", "specificity", "mcc", "accuracy"]
        names = ["Balanced accuracy", "F1", "Precision", "Recall", "Specificity", "MCC", "Accuracy"]
        for _, r in f.iterrows():
            rows.append({"Task": TASK_NAME[task], "Feature selection": STRATEGY[m["best_strategy"]],
                         "Fold": int(r.fold), **{n: r[c] for c, n in zip(cols, names)}})
        rows.append({"Task": TASK_NAME[task], "Feature selection": STRATEGY[m["best_strategy"]],
                     "Fold": "mean", **{n: f[c].mean() for c, n in zip(cols, names)}})
    out.append(("Table S12", "Operating performance of the task-level MERGE at a score threshold of 0.5, per outer fold "
                "and as the fold mean (Table 6)", r4(pd.DataFrame(rows))))
    # S13 paired tests
    rows = []
    for _, r in read_csv(os.path.join(RESULTS, "stats", "paired_tests.csv")).iterrows():
        rows.append({"Task": TASK_NAME[r.task], "Comparison": r.comparison, "Reference": pretty(r.reference),
                     "Method": "pooled out-of-fold; DeLong test and 2,000 class-stratified bootstrap replicates",
                     "Variants": r.n, "Positives": r.positives, "MERGE AUROC": r.merge_auroc,
                     "Reference AUROC": r.reference_auroc, "ΔAUROC": r.delta_auroc, "ΔAUROC 95% CI low": r.delta_auroc_ci_lo,
                     "ΔAUROC 95% CI high": r.delta_auroc_ci_hi, "DeLong P": r.delong_p,
                     "MERGE AUPRC": r.merge_auprc, "Reference AUPRC": r.reference_auprc, "ΔAUPRC": r.delta_auprc,
                     "ΔAUPRC 95% CI low": r.delta_auprc_ci_lo, "ΔAUPRC 95% CI high": r.delta_auprc_ci_hi,
                     "ΔAUPRC bootstrap P": r.delta_auprc_boot_p})
    for c in json.load(open(os.path.join(RESULTS, "stats_cluster", "cluster_bootstrap.json"))):
        ref = meta(c["task"])["best_single_feature"]
        rows.append({"Task": TASK_NAME[c["task"]], "Comparison": "strongest individual feature", "Reference": pretty(ref),
                     "Method": f"pooled out-of-fold; chromosome-cluster bootstrap over {c['n_chr']} chromosomes",
                     "Variants": c["n"], "Positives": np.nan, "MERGE AUROC": c["merge_auroc"],
                     "Reference AUROC": c["single_auroc"], "ΔAUROC": c["delta_auroc"],
                     "ΔAUROC 95% CI low": c["delta_auroc_ci_cluster"][0], "ΔAUROC 95% CI high": c["delta_auroc_ci_cluster"][1],
                     "DeLong P": np.nan, "MERGE AUPRC": c["merge_auprc"], "Reference AUPRC": c["single_auprc"],
                     "ΔAUPRC": c["delta_auprc"], "ΔAUPRC 95% CI low": c["delta_auprc_ci_cluster"][0],
                     "ΔAUPRC 95% CI high": c["delta_auprc_ci_cluster"][1], "ΔAUPRC bootstrap P": c["delta_auprc_p_cluster"]})
    s13 = pd.DataFrame(rows)
    out.append(("Table S13", "Paired comparisons of MERGE with named references: pooled DeLong and class-stratified "
                "bootstrap (Table 3), and chromosome-cluster bootstrap against the strongest individual feature",
                r4(s13, [c for c in s13.columns if s13[c].dtype.kind == "f" and c not in ("DeLong P", "ΔAUPRC bootstrap P")])))
    # S14 missing scores
    rows = []
    for task, fname in TASKS.items():
        d = read_csv(os.path.join(DATA, fname), low_memory=False)
        d = d[d["chr"].map(fold_of) != -1]
        for col in read_csv(os.path.join(RESULTS, f"{task}_singlefeature.csv")).feature:
            miss = int(d[col].isna().sum())
            rows.append({"Task": TASK_NAME[task], "Feature": pretty(col), "Column": col, "Variants": len(d),
                         "Missing": miss, "Missing (%)": round(100 * miss / len(d), 2),
                         "Handling": ("training-fold median, then label-free percentile against the training folds; "
                                      "never a MERGE input" if is_ag(col) else
                                      "training-fold median inside the MERGE pipeline; the deployed models store the "
                                      "training median")})
    out.append(("Table S14", "Missing scores per feature and how they were handled", pd.DataFrame(rows)))
    # S15 splicing out-of-fold predictions
    mo = read_csv(os.path.join(RESULTS, "splicing_merge_oof.csv"))
    av = read_csv(os.path.join(AGAVG, "splicing_alphagenome_average_oof.csv"))
    assert len(mo) == len(av) == 35015 and (mo[["pos", "ref", "alt"]].values == av[["pos", "ref", "alt"]].values).all()
    best = meta("splicing")["best_strategy"]
    s15 = pd.DataFrame({"chr": av["chr"], "pos": av["pos"], "ref": av["ref"], "alt": av["alt"], "label": av["label"],
                        "fold": av["fold"], f"MERGE ({STRATEGY[best]}, task level)": mo[best],
                        "MERGE (LASSO)": mo["lasso"], "MERGE (correlation filtering)": mo["correlation"],
                        "MERGE (beam search)": mo["beam"],
                        "AlphaGenome splicing (raw; empty = missing)": av["alphagenome_raw"].where(av["alphagenome_observed"] == 1),
                        "AlphaGenome observed": av["alphagenome_observed"],
                        "AlphaGenome percentile (training folds)": av["alphagenome_component"],
                        "MERGE + AlphaGenome average": av["merge_alphagenome_average"]})
    assert np.allclose(s15[f"MERGE ({STRATEGY[best]}, task level)"], av["merge_no_alphagenome"])
    out.append(("Table S15", "Variant-level out-of-fold predictions for all 35,015 splicing variants", r4(s15, nd=6)))
    return out


# ───────────────────────── Additional file 3 ─────────────────────────

GROUP = {"MERGE": "MERGE", "A": "A: not trained on clinical pathogenicity labels",
         "B": "B: trained on clinical pathogenicity labels"}


def pname(p):
    return {"AlphaGenome_raw_max": "AlphaGenome raw maximum", "AlphaGenome_quantile_max": "AlphaGenome quantile maximum",
            "phyloP100way": "phyloP 100-way", "phyloP100way_dbnsfp": "phyloP 100-way (dbNSFP)",
            "phyloP470way_dbnsfp": "phyloP 470-way (dbNSFP)", "CADD_dbnsfp": "CADD (dbNSFP)",
            "Polyphen2_HVAR": "PolyPhen-2 HVAR", "ReMM_v04": "ReMM v0.4", "ReMM_v031": "ReMM v0.3.1",
            "frameshift_rule": "Frameshift rule"}.get(p, pretty(p))


def file3():
    out = []
    b = read_csv(os.path.join(BASE, "baseline_comparison.csv"))
    b["Task"] = b.task.map(TASK_NAME); b["Predictor"] = b.predictor.map(pname)
    b["Training labels"] = b.tier.map(lambda t: GROUP.get(t, "fixed average of MERGE and AlphaGenome"))
    s16 = b[["Task", "Predictor", "Training labels", "n_common", "positives_common", "auroc_common", "auroc_sd_common",
             "auprc_common", "auprc_sd_common"]].rename(columns={
        "n_common": "Common subset", "positives_common": "Positives", "auroc_common": "AUROC, fold mean",
        "auroc_sd_common": "AUROC, sample SD", "auprc_common": "AUPRC, fold mean", "auprc_sd_common": "AUPRC, sample SD"})
    out.append(("Table S16", "Established predictors, MERGE and the fixed average on the subset of variants scored by "
                "every predictor of the task (common subset)", r4(s16.sort_values(["Task", "AUROC, fold mean"], ascending=[True, False]))))
    s17 = b[["Task", "Predictor", "Training labels", "n_scored", "coverage", "auroc_own", "auroc_sd_own",
             "auprc_own", "auprc_sd_own"]].rename(columns={
        "n_scored": "Variants scored", "coverage": "Coverage", "auroc_own": "AUROC, fold mean (own scored subset)",
        "auroc_sd_own": "AUROC, sample SD", "auprc_own": "AUPRC, fold mean (own scored subset)", "auprc_sd_own": "AUPRC, sample SD"})
    out.append(("Table S17", "Coverage of each predictor and its performance on all variants it scores", r4(s17)))
    s = read_csv(os.path.join(BASE, "significance.csv"))
    s18 = pd.DataFrame({"Task": s.task.map(TASK_NAME), "Score": s.anchor, "Comparator": s.comparator.map(pname),
                        "Variants scored by both": s.n, "Positives": s.positives, "ΔAUROC": s.d_auroc,
                        "ΔAUROC 95% CI low": s.auroc_lo, "ΔAUROC 95% CI high": s.auroc_hi,
                        "ΔAUROC interval excludes 0": s.auroc_resolved, "ΔAUPRC": s.d_auprc,
                        "ΔAUPRC 95% CI low": s.auprc_lo, "ΔAUPRC 95% CI high": s.auprc_hi,
                        "ΔAUPRC interval excludes 0": s.auprc_resolved})
    out.append(("Table S18", "Paired differences from established predictors with chromosome-cluster bootstrap intervals "
                "(Table 8); each pair uses the variants scored by both", r4(s18)))
    g = read_csv(os.path.join(BASE, "baseline_comparison_by_region.csv"))
    s19 = pd.DataFrame({"Task": g.task.map(TASK_NAME), "Region": g.region, "Predictor": g.predictor.map(pname),
                        "Variants": g.n, "Positives": g.positives, "AUROC": g.auroc, "AUPRC": g.auprc})
    out.append(("Table S19", "Noncoding variants by region class (ANNOVAR): pooled AUROC and AUPRC", r4(s19)))
    flow = read_csv(os.path.join(SGE, "cohort", "sge_cohort_flow.csv"))
    sm = json.load(open(os.path.join(SGE, "results", "sge_modeling_meta.json")))
    labels = {"official_table": "Variants in Findlay et al. (2018) Supplementary Table 1",
              "missense_LOF_FUNC_source_ClinVar_absent": "Missense SNVs classified LOF or FUNC, not in ClinVar at the source time",
              "valid_hg19_snv": "Valid hg19 SNV", "unique_hg38_liftover": "Lifted over to hg38 (UCSC chain)",
              "hg38_reference_match": "hg38 reference base matches", "unique_hg38_variant": "Unique hg38 variant",
              "absent_from_local_ClinVar_20251221": "Of these, absent from the 21 December 2025 ClinVar release "
                                                    "(recorded, not used as a filter)",
              "absent_from_original_coding_training": "Validation cohort: absent from the coding training table"}
    s20 = pd.DataFrame({"Step": flow.stage.map(lambda x: labels.get(x, x)), "Variants": flow.n})
    extra = pd.DataFrame({"Step": ["Final cohort: loss of function (positives)", "Common subset scored by every predictor",
                                   "Common subset: loss of function"],
                          "Variants": [sm["positives"], sm["common_subset"], sm["common_subset_positives"]]})
    out.append(("Table S20", "BRCA1 saturation genome editing cohort", pd.concat([s20, extra], ignore_index=True)))
    mt = read_csv(os.path.join(SGE, "results", "sge_external_validation_metrics.csv"))
    s21 = pd.DataFrame({"Predictor": mt.predictor.map(lambda p: "MERGE" if p.startswith("MERGE") else pname(p)),
                        "Kind": mt.kind, "Variants (common subset)": mt.n_common, "Coverage of the full cohort": mt.coverage_full,
                        "AUROC": mt.auroc, "AUROC 95% CI low": mt.auroc_ci_low, "AUROC 95% CI high": mt.auroc_ci_high,
                        "AUPRC": mt.auprc, "AUPRC 95% CI low": mt.auprc_ci_low, "AUPRC 95% CI high": mt.auprc_ci_high})
    out.append(("Table S21", f"All {len(mt)} predictors on the BRCA1 SGE common subset (variant-level bootstrap intervals, "
                "2,000 replicates)", r4(s21)))
    dl = read_csv(os.path.join(SGE, "results", "sge_paired_bootstrap_deltas.csv"))
    s22 = pd.DataFrame({"Comparator": dl.comparison.str.replace("MERGE without AlphaGenome - ", "", regex=False).map(pname),
                        "ΔAUROC (MERGE - comparator)": dl.delta_auroc, "ΔAUROC 95% CI low": dl.delta_auroc_ci_low,
                        "ΔAUROC 95% CI high": dl.delta_auroc_ci_high, "P (ΔAUROC)": dl.p_auroc,
                        "ΔAUPRC": dl.delta_auprc, "ΔAUPRC 95% CI low": dl.delta_auprc_ci_low,
                        "ΔAUPRC 95% CI high": dl.delta_auprc_ci_high, "P (ΔAUPRC)": dl.p_auprc})
    out.append(("Table S22", "Paired bootstrap differences between MERGE and every other predictor on the BRCA1 SGE "
                "common subset", r4(s22, [c for c in s22.columns if s22[c].dtype.kind == "f" and not c.startswith("P (")])))
    pw = read_csv(os.path.join(SGE, "results", "paired_with_vs_without_BRCA1.csv")).iloc[0]
    nb = read_csv(os.path.join(SGE, "results", "sge_noBRCA1_metrics.csv")).iloc[0]
    nm = json.load(open(os.path.join(SGE, "results", "sge_noBRCA1_meta.json")))["MERGE without AlphaGenome"]
    full = mt[mt.predictor.str.startswith("MERGE")].iloc[0]
    s23 = pd.DataFrame([
        {"Model": "MERGE-Coding (released)", "Training variants": 4289, "Algorithm": pretty_algo(sm["merge_algorithm"]),
         "Features": "; ".join(pretty(x) for x in sm["merge_features"]), "Evaluation variants": int(full.n_common),
         "AUROC": full.auroc, "AUROC 95% CI low": full.auroc_ci_low, "AUROC 95% CI high": full.auroc_ci_high,
         "AUPRC": full.auprc, "AUPRC 95% CI low": full.auprc_ci_low, "AUPRC 95% CI high": full.auprc_ci_high},
        {"Model": "MERGE-Coding retrained without BRCA1", "Training variants": nm["n_train"],
         "Algorithm": pretty_algo(nm["algorithm"]), "Features": "; ".join(pretty(x) for x in nm["features"]),
         "Evaluation variants": int(pw.n_common), "AUROC": pw.auroc_no_BRCA1, "AUPRC": pw.auprc_no_BRCA1,
         "ΔAUROC vs released": pw.d_auroc, "ΔAUROC 95% CI low": pw.d_auroc_ci_low, "ΔAUROC 95% CI high": pw.d_auroc_ci_high,
         "P (ΔAUROC)": pw.auroc_p, "ΔAUPRC vs released": pw.d_auprc, "ΔAUPRC 95% CI low": pw.d_auprc_ci_low,
         "ΔAUPRC 95% CI high": pw.d_auprc_ci_high, "P (ΔAUPRC)": pw.auprc_p},
        {"Model": "MERGE-Coding retrained without BRCA1 (full cohort)", "Training variants": nm["n_train"],
         "Algorithm": pretty_algo(nm["algorithm"]), "Features": "; ".join(pretty(x) for x in nm["features"]),
         "Evaluation variants": int(nb.n_common), "AUROC": nb.auroc, "AUROC 95% CI low": nb.auroc_ci_low,
         "AUROC 95% CI high": nb.auroc_ci_high, "AUPRC": nb.auprc, "AUPRC 95% CI low": nb.auprc_ci_low,
         "AUPRC 95% CI high": nb.auprc_ci_high}])
    out.append(("Table S23", "Gene-level training exclusion: MERGE-Coding retrained without the 212 BRCA1 variants "
                "(chr17:43,044,295-43,125,364) and evaluated on the SGE cohort",
                r4(s23, [c for c in s23.columns if s23[c].dtype.kind == "f" and not c.startswith("P (")])))
    return out


# ───────────────────────── writing ─────────────────────────

NOTES = [
    "Folds: three chromosome-held-out outer folds. " + "; ".join(f"fold {k}: chr {v}" for k, v in FOLD_CHROMS.items()) + ".",
    "Fold means are unweighted means over the three outer folds; SDs are sample SDs (divisor 2). Pooled metrics rank "
    "out-of-fold predictions across folds and differ from fold means.",
    "AlphaGenome is never a MERGE input. As an individual score it is evaluated by a label-free percentile against the "
    "training folds after training-fold median imputation; it enters the fixed MERGE + AlphaGenome average only.",
    "Source: MERGE-benchmark repository, written by 04_modeling/additional_files.py from the committed results.",
]


def write(path, title, description, tables):
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        contents = pd.DataFrame({"Table": [t for t, _, _ in tables], "Content": [c for _, c, _ in tables],
                                 "Rows": [len(d) for _, _, d in tables]})
        head = pd.DataFrame({"Table": [title, description, ""] + ["Note"] * len(NOTES) + [""],
                             "Content": ["", "", ""] + NOTES + [""], "Rows": [""] * (4 + len(NOTES))})
        pd.concat([head, contents], ignore_index=True).to_excel(xw, sheet_name="Contents", index=False, header=False)
        for name, caption, df in tables:
            df.to_excel(xw, sheet_name=name, index=False, startrow=2)
            xw.sheets[name].cell(row=1, column=1, value=f"{name}. {caption}.").font = Font(bold=True)
    wb = load_workbook(path)
    for ws in wb.worksheets:
        if ws.title == "Contents":
            ws["A1"].font = Font(bold=True, size=13)
            ws.column_dimensions["A"].width = 14; ws.column_dimensions["B"].width = 140
            for row in ws.iter_rows():
                for c in row:
                    c.alignment = Alignment(wrap_text=True, vertical="top")
            continue
        for c in ws[3]:
            c.font = Font(bold=True); c.alignment = Alignment(wrap_text=True, vertical="top")
        ws.freeze_panes = "A4"
        for col in ws.iter_cols(min_row=3, max_row=min(ws.max_row, 300)):
            width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
            ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 60)
    wb.save(path)


def main():
    outdir = sys.argv[1]
    os.makedirs(outdir, exist_ok=True)
    files = [
        ("Additional_file_1.xlsx", "Additional file 1. Individual model benchmarks",
         "Summary and fold-specific metrics for all 25 coding, 18 splicing, 23 noncoding and 22 non-SNV features "
         "(Tables S1-S2). All splicing comparisons use the fixed cohort of 35,015 variants.", file1()),
        ("Additional_file_2.xlsx", "Additional file 2. MERGE integration and assessment",
         "Feature-selection strategies, fixed MERGE-AlphaGenome averaging, outer-fold results, class-balance analyses, "
         "selected panels, inner-validation algorithm comparisons, SHAP attribution, calibration, operating points and "
         "paired tests (Tables S3-S13); missing-score handling and variant-level out-of-fold predictions for all 35,015 "
         "splicing variants (Tables S14-S15). Every supervised panel excludes AlphaGenome.", file2()),
        ("Additional_file_3.xlsx", "Additional file 3. External comparisons and BRCA1 validation",
         "Established-predictor performance and coverage, paired chromosome-cluster comparisons, noncoding regional "
         "analyses, BRCA1 saturation genome editing results and gene-level training exclusion (Tables S16-S23).", file3()),
    ]
    for fname, title, desc, tables in files:
        write(os.path.join(outdir, fname), title, desc, tables)
        print(fname, ", ".join(f"{t} ({len(d)})" for t, _, d in tables))


if __name__ == "__main__":
    main()
