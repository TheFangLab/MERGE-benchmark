# 05_results — benchmark results

Everything here is produced by `04_modeling/run_all.sh` from `03_datasets/`.
The manuscript tables are in `tables/` (`TABLES.md` for reading, one CSV per table).

MERGE is trained on the candidate pool with every `alphagenome_*` column removed.
AlphaGenome appears only as an independent comparator in the individual-feature files,
scored by a label-free percentile, and in `../05_results_alphagenome_average/`.

## Files

| File | Content |
|---|---|
| `{task}_merge_oof.csv` | Out-of-fold MERGE probability per variant, one column per feature-selection strategy (`lasso`, `correlation`, `beam`) |
| `{task}_merge_folds.csv` | Per outer fold and strategy: selected algorithm, panel size, AUROC, AUPRC and the 0.5-threshold metrics |
| `{task}_merge_summary.csv` | The same, mean and sample SD over the three folds, per strategy |
| `{task}_meta.json` | Cohort size, selected panels per fold and strategy, task-level strategy, strongest feature overall and strongest MERGE candidate input |
| `{task}_algo_inner.csv` | Inner-CV AUROC of the ten algorithms on each fold's LASSO panel (Fig. 4) |
| `{task}_shap.csv`, `{task}_shap_beeswarm.csv` | Mean \|SHAP\| and the held-out beeswarm sample for the task-level panels (Fig. 5) |
| `{task}_singlefeature.csv`, `{task}_singlefeature_oof.csv` | Every benchmark feature: fold mean ± sample SD, and per-variant out-of-fold scores (Table 1, Figs 2–3) |
| `stats/paired_tests.{csv,json}` | MERGE minus named references on pooled out-of-fold predictions (Table 3) |
| `calibration/calibration.json` | Brier score, ECE and reliability bins (Table 5) |
| `stats_cluster/cluster_bootstrap.json` | Chromosome-cluster intervals for MERGE minus the strongest single feature |
| `balanced_1to1/` | The same pipeline after majority-class downsampling to 1:1 (Table 4, Fig. 6) |
| `tables/` | Manuscript Tables 1–10 |
| `trained_models/` | Deployable models refitted on every variant, and their model card |

`{task}` is `coding`, `splicing`, `noncoding` or `nonsnv`. Rows of the out-of-fold files
follow the dataset table in `03_datasets/`, restricted to the chromosomes of the three
folds (all variants).

## Task-level MERGE

| Task | n | Strategy | AUROC | AUPRC |
|---|---:|---|---:|---:|
| Coding SNVs | 4,289 | beam search | 0.9519 ± 0.0272 | 0.9729 ± 0.0171 |
| Splicing SNVs | 35,015 | LASSO | 0.9269 ± 0.0042 | 0.9561 ± 0.0027 |
| Noncoding SNVs | 86,829 | beam search | 0.9472 ± 0.0137 | 0.6841 ± 0.0258 |
| Non-SNVs | 36,862 | LASSO | 0.9820 ± 0.0035 | 0.9943 ± 0.0017 |

Mean ± sample SD over the three chromosome-held-out outer folds.

## Deployable models

`trained_models/MERGE_noAlphaGenome_{task}.pkl` are the models served by the web server
and `merge-cli`. Each is a joblib dict; `bundle["pipeline"].predict_proba(X[bundle["features"]])[:, 1]`
gives the score, with median imputation and standardization inside the pipeline. They
are fitted on every variant without a held-out split, so their fit is not a performance
estimate; use the out-of-fold results above.
