# 06_figures — manuscript figures

Each figure is written as PNG (screen) and as LZW-compressed 300-dpi TIFF without an alpha
channel (submission). Figures carry no titles; captions are in the manuscript.

| File | Manuscript | Produced by | From |
|---|---|---|---|
| `fig1_overview` | Fig. 1, study overview | schematic (no script) | — |
| `fig2_auroc` | Fig. 2, AUROC of individual features, MERGE and the fixed average | `04_modeling/make_figures.py` | `05_results/`, `05_results_alphagenome_average/` |
| `fig3_auprc` | Fig. 3, AUPRC, same layout | `04_modeling/make_figures.py` | as Fig. 2 |
| `fig4_algorithms_per_dataset` | Fig. 4, inner-CV AUROC of the ten algorithms | `04_modeling/make_figures2.py` | `05_results/{task}_algo_inner.csv` |
| `fig5_shap` | Fig. 5, held-out SHAP beeswarms | `04_modeling/make_figures.py` | `05_results/{task}_shap_beeswarm.csv` |
| `fig6_class_balance` | Fig. 6, original vs 1:1 class ratio | `04_modeling/make_figures2.py` | `05_results/`, `05_results/balanced_1to1/` |
| `fig7_baselines_auroc` | Fig. 7, AUROC vs established predictors | `07_baselines/scripts/make_baseline_figures.py` | `07_baselines/results/baseline_comparison.csv` |
| `fig8_baselines_auprc` | Fig. 8, AUPRC vs established predictors | `07_baselines/scripts/make_baseline_figures.py` | as Fig. 7 |
| `fig9_sge_external_validation` | Fig. 9, BRCA1 saturation genome editing | `08_external_validation/scripts/06_make_sge_figure.py` | `08_external_validation/results/` |
