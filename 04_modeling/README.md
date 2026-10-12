# 04_modeling — benchmark, supervised integration, statistics, tables and figures

```bash
pip install -r environment/requirements-modeling.txt     # Python 3.11
bash 04_modeling/run_all.sh                               # everything, in place
```

`run_all.sh` regenerates every result, table and figure from `03_datasets/` in
dependency order (about two hours on a 16-core CPU). Paths default to the directories
of this repository (`config.py`); outputs overwrite the committed files, so `git diff`
shows what changed.

| File | What it does | Manuscript |
|---|---|---|
| `config.py` | Folds, task tables, seed 42, the 95% coverage rule, and the removal of every `alphagenome_*` column from the MERGE candidate pool | Methods |
| `pipeline.py` | One task end to end: the three feature-selection strategies, the ten-algorithm pool chosen by inner chromosome-grouped CV, outer chromosome-held-out predictions, SHAP. `balanced` repeats it after 1:1 downsampling | Table 2, 4, 6; Figs 4–6 |
| `singlefeature_benchmark.py` | Every benchmark feature on the same folds: single-feature logistic regression, or for AlphaGenome a label-free training-fold percentile after training-fold median imputation | Table 1; Figs 2–3 |
| `assemble_final_results.py` | Records the strongest feature overall and the strongest MERGE candidate input in `{task}_meta.json` | — |
| `stats_tests.py` | Paired comparisons on pooled out-of-fold predictions: DeLong test and 2,000 class-stratified bootstrap resamples | Table 3 |
| `fast_metrics.py` | AUROC / average precision from bootstrap counts (identical to scikit-learn, checked at run time) | — |
| `calibration_and_cluster_bootstrap.py` | Brier score, ECE and reliability bins; chromosome-cluster intervals for MERGE minus the strongest single feature | Table 5; Results text |
| `alphagenome_average.py` | The fixed MERGE + AlphaGenome average for splicing and noncoding variants, and the deployable AlphaGenome reference table | Table 2 |
| `calibrate_alphagenome_average_thresholds.py` | Interpretation cut-points for the average, matched to the MERGE bands without labels | Methods (web server) |
| `export_models.py` | Refits each task's model on every variant, the procedure behind the released models in `05_results/trained_models/` (writes `trained_models_refit/`) | Methods |
| `manuscript_tables.py` | Tables 1–10 (`05_results/tables/`) | Tables 1–10 |
| `additional_files.py` | Additional files 1–3 (Tables S1–S23) as `.xlsx`, from the committed results; checks its per-fold single-feature metrics against `{task}_singlefeature.csv` | Additional files |
| `make_figures.py`, `make_figures2.py` | Figures 2, 3, 5 and 4, 6 | Figs 2–6 |
| `figstyle.py` | Shared palette, display names and figure output | — |

## Design

* **Nothing crosses folds.** Median imputation, standardization, feature selection and
  algorithm choice are fitted on the training folds only; inner validation is
  `GroupKFold` grouped by chromosome. The AlphaGenome percentile also takes its
  reference distribution, and the median used for missing scores, from the training
  folds.
* **MERGE never sees AlphaGenome output.** The columns are removed before selection,
  and `export_models.py` refuses to write a model whose panel contains one.
  AlphaGenome is evaluated as an independent comparator and through the fixed
  average, neither of which fits anything to labels.
* **Task-level MERGE** is the feature-selection strategy with the highest mean outer-fold
  AUROC: beam search (coding), LASSO (splicing, noncoding, non-SNV). All three
  strategies are kept in `{task}_merge_oof.csv` and
  `{task}_merge_summary.csv`.
* **Fold summaries** are unweighted means and sample SDs (divisor 2) over the three
  outer folds. Pooled metrics (Tables 3 and 5) rank predictions across folds and differ
  from the fold means.
* **Bootstrap streams** are seeded per comparison, so each interval is independent of
  which other comparisons are run.
