"""
Fit the deployable MERGE bundle for one or more tasks, on the candidate pool
with every `alphagenome_*` column removed.

Same procedure as the out-of-fold benchmark, applied to all rows: the best
feature-selection strategy is read from that task's `{task}_meta.json`, the
panel is re-selected on the full dataset with that strategy, the algorithm is
re-chosen by inner GroupKFold grouped by chromosome, and the pipeline
(median impute -> z-score -> classifier) is refit on every row.

The bundle records the exact source table and its SHA-256, so a model can always
be traced to the dataset revision it was fitted on.

Refusal, not a warning: the exporter raises if any selected feature starts with
`alphagenome_`, so an AlphaGenome-trained bundle cannot be produced by accident.

Usage: python export_models.py [outdir] [task ...]   (default 05_results/trained_models_refit)
"""
import hashlib
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
import sklearn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS, DATA, TASKS, ENSEMBLE_EXCLUDE_PREFIXES
import pipeline as P

DROP = ENSEMBLE_EXCLUDE_PREFIXES


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def versions():
    import catboost, joblib as jl, lightgbm, xgboost
    return {"python": sys.version.split()[0], "numpy": np.__version__,
            "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
            "joblib": jl.__version__, "xgboost": xgboost.__version__,
            "lightgbm": lightgbm.__version__, "catboost": catboost.__version__}


def fit_bundle(task, strat, path=None):
    """Select the panel with `strat`, choose the algorithm by inner GroupKFold and
    refit on every row of the task table (or of `path`, a table in the same format)."""
    df, feats, dropped, ablated = P.load_task(task, DROP, path=path)
    y = df["label"].values.astype(int)
    groups = df["chr"].values
    sel = P.STRATEGIES[strat](df, y, groups, feats)
    if any(c.startswith("alphagenome") for c in sel):
        raise SystemExit(f"{task}: an alphagenome_* column reached the panel; refusing")

    pool = P.algo_pool(y)
    Xsel = df[sel].values
    scores = {n: P.inner_cv_auroc(lambda e=e: e.__class__(**e.get_params()),
                                  Xsel, y, groups) for n, e in pool.items()}
    best = max(scores, key=lambda k: scores[k] if scores[k] == scores[k] else -1)

    est = pool[best]
    pipe = P.preprocess_pipe(est.__class__(**est.get_params()))
    pipe.fit(Xsel, y)
    return pipe, sel, best, scores, df, feats, dropped, ablated


def export(task, outdir):
    t0 = time.time()
    strat = json.load(open(f"{RESULTS}/{task}_meta.json"))["best_strategy"]
    pipe, sel, best, scores, df, feats, dropped, ablated = fit_bundle(task, strat)
    y = df["label"].values.astype(int)

    src = os.path.realpath(os.path.join(DATA, TASKS[task]))
    bundle = {
        "format_version": 1, "task": task, "pipeline": pipe, "features": sel,
        "eligible_features": feats, "ablated_features": ablated,
        "dropped_features": dropped, "contains_alphagenome_features": False,
        "feature_selection": strat, "algorithm": best,
        "algorithm_inner_group_cv_auroc": {k: (None if v != v else float(v))
                                           for k, v in scores.items()},
        "n_train": int(len(df)), "positive_rate": float(y.mean()),
        "positive_class": 1, "positive_label": "pathogenic / function-altering",
        "training_scope": "full original-ratio dataset; no held-out split",
        "benchmark_note": "Use OOF result files, not training-fit performance, "
                          "for benchmark reporting.",
        "source_data": f"03_datasets/{TASKS[task]}", "source_data_sha256": sha256(src),
        "benchmark_meta": f"05_results/{task}_meta.json",
        "software_versions": versions(),
        "usage": "p = bundle['pipeline'].predict_proba(df[bundle['features']])[:, 1]",
    }
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"MERGE_noAlphaGenome_{task}.pkl")
    joblib.dump(bundle, path, compress=3)
    print(f"[{task}] {path}  strat={strat} algo={best} nfeat={len(sel)} "
          f"n={len(df)} in {time.time() - t0:.0f}s", flush=True)
    return bundle


if __name__ == "__main__":
    outdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RESULTS, "trained_models_refit")
    for t in (sys.argv[2:] or list(TASKS)):
        export(t, outdir)
