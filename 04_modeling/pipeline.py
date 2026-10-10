"""
MERGE benchmark pipeline (one task end-to-end).

Implements the design described in the manuscript Methods:
  - chromosome-held-out 3-fold outer CV, inner GroupKFold for selection
  - preprocessing (median impute + z-score) fit on training fold only
  - single-feature benchmark (per-model ranking)
  - three feature-selection strategies: LASSO, correlation filtering, beam search
  - extended algorithm pool selected within inner CV
  - AUROC / AUPRC + 0.5-threshold metrics, SHAP, OOF predictions saved

The candidate pool excludes every alphagenome_* column (config.ENSEMBLE_EXCLUDE_PREFIXES).

Usage: python pipeline.py <task> [balanced]
"""
import sys, os, json, warnings, time
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, recall_score, accuracy_score,
                             balanced_accuracy_score, matthews_corrcoef,
                             confusion_matrix)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (RANDOM_STATE, DATA, RESULTS, BALANCED, ID_COLS, FOLDS, TASKS,
                    MIN_COVERAGE, ENSEMBLE_EXCLUDE_PREFIXES)

RS = RANDOM_STATE
np.random.seed(RS)


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def balance_1to1(df):
    """Globally downsample the majority class to a 1:1 positive:negative ratio."""
    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]
    n = min(len(pos), len(neg))
    out = pd.concat([pos.sample(n, random_state=RS), neg.sample(n, random_state=RS)])
    return out.sort_values(["chr", "pos"]).reset_index(drop=True)


def load_task(task, drop_prefixes=(), path=None):
    """Load a task table (or `path`, a table in the same format) with its folds."""
    df = pd.read_csv(path or os.path.join(DATA, TASKS[task]), low_memory=False)
    df["chr"] = df["chr"].astype(str).str.replace("chr", "", regex=False)
    feats = [c for c in df.columns if c not in ID_COLS]
    ablated = [c for c in feats if any(c.startswith(p) for p in drop_prefixes)]
    feats = [c for c in feats if c not in ablated]
    # eligibility filter (kept for method consistency; expected to drop nothing)
    keep, dropped = [], []
    for c in feats:
        cov = df[c].notna().mean()
        nuni = df[c].nunique(dropna=True)
        if cov >= MIN_COVERAGE and nuni > 1:
            keep.append(c)
        else:
            dropped.append((c, round(float(cov), 4), int(nuni)))
    df["fold"] = -1
    for k, chroms in FOLDS.items():
        df.loc[df["chr"].isin(chroms), "fold"] = k
    df = df[df["fold"] != -1].reset_index(drop=True)
    return df, keep, dropped, ablated


def preprocess_pipe(clf):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", clf),
    ])


# ----------------------------------------------------------------------------
# algorithm pool  (same across all tasks for comparability)
# SVM-RBF omitted: O(n^2) kernel does not scale to 86k rows.
# ----------------------------------------------------------------------------
def algo_pool(y):
    n_pos = int(np.sum(y == 1)); n_neg = int(np.sum(y == 0))
    spw = max(n_neg, 1) / max(n_pos, 1)
    pool = {
        "LR_L2": LogisticRegression(penalty="l2", class_weight="balanced",
                                    max_iter=3000, random_state=RS),
        "LR_L1": LogisticRegression(penalty="l1", solver="saga",
                                    class_weight="balanced", max_iter=3000,
                                    random_state=RS),
        "ElasticNet": LogisticRegression(penalty="elasticnet", l1_ratio=0.5,
                                         solver="saga", class_weight="balanced",
                                         max_iter=3000, random_state=RS),
        "RandomForest": RandomForestClassifier(n_estimators=400,
                                               class_weight="balanced",
                                               n_jobs=-1, random_state=RS),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=400,
                                           class_weight="balanced",
                                           n_jobs=-1, random_state=RS),
        "HistGB": HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                                 class_weight="balanced",
                                                 random_state=RS),
        "XGBoost": XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                                 subsample=0.9, colsample_bytree=0.9,
                                 scale_pos_weight=spw, eval_metric="logloss",
                                 tree_method="hist", n_jobs=-1, random_state=RS,
                                 verbosity=0),
        "LightGBM": LGBMClassifier(n_estimators=400, learning_rate=0.05,
                                   class_weight="balanced", n_jobs=-1,
                                   random_state=RS, verbose=-1),
        "CatBoost": CatBoostClassifier(iterations=400, learning_rate=0.05,
                                       depth=5, auto_class_weights="Balanced",
                                       random_seed=RS, verbose=0,
                                       allow_writing_files=False),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400,
                             early_stopping=True, random_state=RS),
    }
    return pool


def inner_cv_auroc(clf_factory, X, y, groups, n_splits=3):
    """Mean inner GroupKFold AUROC for an estimator built by clf_factory()."""
    ng = len(np.unique(groups))
    k = min(n_splits, ng)
    if k < 2:
        return np.nan
    gkf = GroupKFold(n_splits=k)
    aucs = []
    for tr, va in gkf.split(X, y, groups):
        if len(np.unique(y[va])) < 2 or len(np.unique(y[tr])) < 2:
            continue
        pipe = preprocess_pipe(clf_factory())
        pipe.fit(X[tr], y[tr])
        p = pipe.predict_proba(X[va])[:, 1]
        aucs.append(roc_auc_score(y[va], p))
    return float(np.mean(aucs)) if aucs else np.nan


# ----------------------------------------------------------------------------
# three feature-selection strategies (fit on a training fold only)
# ----------------------------------------------------------------------------
def sel_lasso(Xdf, y, groups, feats):
    X = Xdf[feats].values
    best_C, best_auc = None, -1
    for C in [0.01, 0.03, 0.1, 0.3, 1.0]:
        fac = lambda C=C: LogisticRegression(penalty="l1", solver="saga", C=C,
                                             class_weight="balanced",
                                             max_iter=3000, random_state=RS)
        a = inner_cv_auroc(fac, X, y, groups)
        if a is not None and a > best_auc:
            best_auc, best_C = a, C
    pipe = preprocess_pipe(LogisticRegression(penalty="l1", solver="saga",
                                              C=best_C, class_weight="balanced",
                                              max_iter=3000, random_state=RS))
    pipe.fit(X, y)
    coef = np.abs(pipe.named_steps["clf"].coef_.ravel())
    order = np.argsort(coef)[::-1]
    sel = [feats[i] for i in order if coef[i] > 1e-8][:12]
    if not sel:
        sel = [feats[int(np.argmax(coef))]]
    return sel


def sel_corr(Xdf, y, groups, feats):
    X = Xdf[feats]
    corrs = {}
    for c in feats:
        v = X[c].fillna(X[c].median())
        corrs[c] = abs(np.corrcoef(v, y)[0, 1]) if v.std() > 0 else 0.0
    ranked = sorted(feats, key=lambda c: corrs[c], reverse=True)
    Xf = X.fillna(X.median())
    cormat = Xf.corr().abs()
    admitted = []
    for c in ranked:
        if all(cormat.loc[c, s] <= 0.9 for s in admitted):
            admitted.append(c)
    best_set, best_auc = admitted[:3], -1
    for size in [3, 5, 8, 10]:
        cand = admitted[:size]
        if not cand:
            continue
        a = inner_cv_auroc(lambda: LogisticRegression(class_weight="balanced",
                                                      max_iter=3000, random_state=RS),
                           Xdf[cand].values, y, groups)
        if a is not None and a > best_auc:
            best_auc, best_set = a, cand
    return best_set


def sel_beam(Xdf, y, groups, feats, beam=3, max_feats=6):
    def score(subset):
        return inner_cv_auroc(lambda: LogisticRegression(class_weight="balanced",
                                                         max_iter=3000, random_state=RS),
                              Xdf[list(subset)].values, y, groups)
    beams = [([f], score([f])) for f in feats]
    beams = sorted([b for b in beams if not np.isnan(b[1])],
                   key=lambda x: x[1], reverse=True)[:beam]
    best = beams[0] if beams else ([feats[0]], 0.0)
    for _ in range(max_feats - 1):
        cand = []
        for subset, _s in beams:
            for f in feats:
                if f in subset:
                    continue
                ns = subset + [f]
                cand.append((ns, score(ns)))
        cand = sorted([c for c in cand if not np.isnan(c[1])],
                      key=lambda x: x[1], reverse=True)
        if not cand:
            break
        beams = cand[:beam]
        if beams[0][1] > best[1]:
            best = beams[0]
    return best[0]


STRATEGIES = {"lasso": sel_lasso, "correlation": sel_corr, "beam": sel_beam}


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------
def thresh_metrics(y, p, thr=0.5):
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "accuracy": accuracy_score(y, yh),
        "balanced_acc": balanced_accuracy_score(y, yh),
        "f1": f1_score(y, yh, zero_division=0),
        "precision": precision_score(y, yh, zero_division=0),
        "recall": recall_score(y, yh, zero_division=0),
        "specificity": spec,
        "mcc": matthews_corrcoef(y, yh),
    }


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def run(task, balanced=False):
    t0 = time.time()
    outdir = BALANCED if balanced else RESULTS
    os.makedirs(outdir, exist_ok=True)
    df, feats, dropped, ablated = load_task(task, ENSEMBLE_EXCLUDE_PREFIXES)
    if balanced:
        df = balance_1to1(df)
    y_all = df["label"].values.astype(int)
    tag = f"{task}{'-bal' if balanced else ''}"
    print(f"[{tag}] n={len(df)} pos_rate={y_all.mean():.4f} "
          f"n_feats={len(feats)} excluded={ablated} dropped={dropped}", flush=True)

    folds = sorted(FOLDS.keys())

    # ---- single-feature benchmark -----------------------------------------
    sf_rows = []
    sf_oof = df[ID_COLS].copy()
    for f in feats:
        oof = np.full(len(df), np.nan)
        per_fold = []
        for k in folds:
            te = df["fold"].values == k
            tr = ~te
            if len(np.unique(y_all[tr])) < 2:
                continue
            pipe = preprocess_pipe(LogisticRegression(class_weight="balanced",
                                                      max_iter=3000, random_state=RS))
            pipe.fit(df.loc[tr, [f]].values, y_all[tr])
            p = pipe.predict_proba(df.loc[te, [f]].values)[:, 1]
            oof[te] = p
            per_fold.append((roc_auc_score(y_all[te], p),
                             average_precision_score(y_all[te], p)))
        aucs = [a for a, _ in per_fold]; aps = [b for _, b in per_fold]
        sf_rows.append({"feature": f,
                        "auroc_mean": np.mean(aucs), "auroc_sd": np.std(aucs, ddof=1),
                        "auprc_mean": np.mean(aps), "auprc_sd": np.std(aps, ddof=1)})
        sf_oof[f] = oof
    sf = pd.DataFrame(sf_rows).sort_values("auroc_mean", ascending=False)
    sf.to_csv(f"{outdir}/{task}_singlefeature.csv", index=False)
    sf_oof.to_csv(f"{outdir}/{task}_singlefeature_oof.csv", index=False)
    best_feat = sf.iloc[0]["feature"]
    print(f"[{task}] best single feature = {best_feat} "
          f"AUROC={sf.iloc[0]['auroc_mean']:.4f} AUPRC={sf.iloc[0]['auprc_mean']:.4f}",
          flush=True)

    # ---- MERGE: three strategies ------------------------------------------
    merge_oof = df[ID_COLS].copy()
    summary_rows = []
    algo_inner_records = []  # (algo, mean_inner_auroc) for Fig 2
    sel_record = {s: {} for s in STRATEGIES}

    for strat_name, strat_fn in STRATEGIES.items():
        oof = np.full(len(df), np.nan)
        fold_metrics = []
        for k in folds:
            te = df["fold"].values == k
            tr = ~te
            Xtr = df.loc[tr]
            ytr = y_all[tr]
            gtr = df.loc[tr, "chr"].values
            if len(np.unique(ytr)) < 2:
                continue
            sel = strat_fn(Xtr, ytr, gtr, feats)
            # choose algorithm by inner CV AUROC on selected panel
            pool = algo_pool(ytr)
            Xsel = Xtr[sel].values
            best_algo, best_a = None, -1
            for name, est in pool.items():
                fac = (lambda e=est: e.__class__(**e.get_params()))
                a = inner_cv_auroc(fac, Xsel, ytr, gtr)
                if strat_name == list(STRATEGIES)[0]:
                    algo_inner_records.append({"algorithm": name, "auroc": a,
                                               "fold": k})
                if a is not None and a > best_a:
                    best_a, best_algo = a, name
            est = pool[best_algo]
            pipe = preprocess_pipe(est.__class__(**est.get_params()))
            pipe.fit(Xsel, ytr)
            p = pipe.predict_proba(df.loc[te, sel].values)[:, 1]
            oof[te] = p
            m = {"strategy": strat_name, "fold": k, "algorithm": best_algo,
                 "n_selected": len(sel),
                 "auroc": roc_auc_score(y_all[te], p),
                 "auprc": average_precision_score(y_all[te], p)}
            m.update(thresh_metrics(y_all[te], p))
            fold_metrics.append(m)
            sel_record[strat_name][k] = sel
            print(f"[{task}] {strat_name} fold{k} algo={best_algo} "
                  f"nsel={len(sel)} AUROC={m['auroc']:.4f} AUPRC={m['auprc']:.4f}",
                  flush=True)
        merge_oof[strat_name] = oof
        summary_rows.extend(fold_metrics)

    summ = pd.DataFrame(summary_rows)
    summ.to_csv(f"{outdir}/{task}_merge_folds.csv", index=False)
    merge_oof.to_csv(f"{outdir}/{task}_merge_oof.csv", index=False)

    agg = (summ.groupby("strategy")
           .agg(auroc_mean=("auroc", "mean"), auroc_sd=("auroc", "std"),
                auprc_mean=("auprc", "mean"), auprc_sd=("auprc", "std"),
                acc=("accuracy", "mean"), bacc=("balanced_acc", "mean"),
                f1=("f1", "mean"), precision=("precision", "mean"),
                recall=("recall", "mean"), specificity=("specificity", "mean"),
                mcc=("mcc", "mean"))
           .reset_index().sort_values("auroc_mean", ascending=False))
    agg.to_csv(f"{outdir}/{task}_merge_summary.csv", index=False)
    best_strat = agg.iloc[0]["strategy"]
    print(f"[{task}] best strategy = {best_strat} "
          f"AUROC={agg.iloc[0]['auroc_mean']:.4f} AUPRC={agg.iloc[0]['auprc_mean']:.4f}",
          flush=True)

    pd.DataFrame(algo_inner_records).to_csv(
        f"{outdir}/{task}_algo_inner.csv", index=False)

    # ---- SHAP on best strategy (LightGBM surrogate on selected panel) ------
    try:
        import shap
        shap_vals_all, feat_val_all, feat_name_all = [], [], []
        mean_abs = {}
        for k in folds:
            sel = sel_record[best_strat].get(k)
            if not sel:
                continue
            te = df["fold"].values == k
            tr = ~te
            imp = SimpleImputer(strategy="median").fit(df.loc[tr, sel])
            Xtr = imp.transform(df.loc[tr, sel]); Xte = imp.transform(df.loc[te, sel])
            m = LGBMClassifier(n_estimators=300, learning_rate=0.05,
                               class_weight="balanced", n_jobs=-1,
                               random_state=RS, verbose=-1).fit(Xtr, y_all[tr])
            expl = shap.TreeExplainer(m)
            sv = expl.shap_values(Xte)
            if isinstance(sv, list):
                sv = sv[1]
            for j, fn in enumerate(sel):
                mean_abs.setdefault(fn, []).append(np.abs(sv[:, j]).mean())
            # subsample for beeswarm
            idx = np.random.RandomState(RS).choice(
                Xte.shape[0], size=min(1500, Xte.shape[0]), replace=False)
            for j, fn in enumerate(sel):
                shap_vals_all.extend(sv[idx, j].tolist())
                feat_val_all.extend(Xte[idx, j].tolist())
                feat_name_all.extend([fn] * len(idx))
        sh = pd.DataFrame({"feature": list(mean_abs.keys()),
                           "mean_abs_shap": [np.mean(v) for v in mean_abs.values()]}
                          ).sort_values("mean_abs_shap", ascending=False)
        sh.to_csv(f"{outdir}/{task}_shap.csv", index=False)
        pd.DataFrame({"feature": feat_name_all, "shap": shap_vals_all,
                      "value": feat_val_all}).to_csv(
            f"{outdir}/{task}_shap_beeswarm.csv", index=False)
        print(f"[{task}] SHAP done", flush=True)
    except Exception as e:
        print(f"[{task}] SHAP failed: {e}", flush=True)

    meta = {"task": task, "balanced": bool(balanced),
            "n": int(len(df)), "pos_rate": float(y_all.mean()),
            "n_features": len(feats), "excluded_features": ablated,
            "dropped": dropped,
            "best_single_feature": best_feat,
            "best_single_auroc": float(sf.iloc[0]["auroc_mean"]),
            "best_single_auprc": float(sf.iloc[0]["auprc_mean"]),
            "best_strategy": best_strat,
            "merge_auroc": float(agg.iloc[0]["auroc_mean"]),
            "merge_auprc": float(agg.iloc[0]["auprc_mean"]),
            "selected_features": {s: {str(k): v for k, v in d.items()}
                                  for s, d in sel_record.items()},
            "runtime_sec": round(time.time() - t0, 1)}
    with open(f"{outdir}/{task}_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[{task}] DONE in {meta['runtime_sec']}s", flush=True)


if __name__ == "__main__":
    # usage: pipeline.py <task> [balanced]
    run(sys.argv[1], balanced="balanced" in sys.argv[2:])
