#!/usr/bin/env bash
# Regenerate every modelling result, manuscript table and figure from 03_datasets/.
#
#   bash 04_modeling/run_all.sh
#
# Outputs are written in place (05_results/, 05_results_alphagenome_average/,
# 07_baselines/results/, 08_external_validation/results/, 06_figures/), so
# `git diff` shows anything that changed. About two hours on a 16-core CPU, most of
# it in step 1. Environment: environment/requirements-modeling.txt.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/.." && pwd)"
PY="${PYTHON:-python}"
export PYTHONUNBUFFERED=1 OMP_WAIT_POLICY=passive
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS="${OMP_NUM_THREADS:-8}" \
       OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS:-8}"
cd "${HERE}"
TASKS="coding splicing noncoding nonsnv"

echo "== 1. MERGE: feature selection, algorithm choice, out-of-fold predictions, SHAP"
for t in ${TASKS}; do "${PY}" pipeline.py "${t}"; done
for t in ${TASKS}; do "${PY}" pipeline.py "${t}" balanced; done      # Table 4, Fig. 6

echo "== 2. individual features (AlphaGenome by label-free percentile)"
"${PY}" singlefeature_benchmark.py "${REPO}/05_results"
"${PY}" assemble_final_results.py

echo "== 3. paired tests (Table 3), calibration and chromosome-cluster intervals (Table 5)"
"${PY}" stats_tests.py
"${PY}" calibration_and_cluster_bootstrap.py

echo "== 4. MERGE + AlphaGenome average (Table 2) and its interpretation cut-points"
"${PY}" alphagenome_average.py
"${PY}" calibrate_alphagenome_average_thresholds.py

echo "== 5. deployable models, refitted on every variant"
# Written next to, not over, the released models in 05_results/trained_models/:
# tree learners differ slightly across hardware (environment/README.md).
"${PY}" export_models.py "${REPO}/05_results/trained_models_refit"

echo "== 6. established predictors (Tables 7-8) and BRCA1 SGE (Table 9)"
"${PY}" "${REPO}/07_baselines/scripts/extract_alphagenome_feature.py"
"${PY}" "${REPO}/07_baselines/scripts/evaluate.py"
"${PY}" "${REPO}/07_baselines/scripts/merge_vs_baseline_ci.py" --table8
"${PY}" "${REPO}/08_external_validation/scripts/04_evaluate_sge.py"
"${PY}" "${REPO}/08_external_validation/scripts/05_retrain_without_brca1.py"

echo "== 7. manuscript tables and figures"
"${PY}" manuscript_tables.py
"${PY}" make_figures.py                                                   # Figs 2, 3, 5
"${PY}" make_figures2.py                                                  # Figs 4, 6
"${PY}" "${REPO}/07_baselines/scripts/make_baseline_figures.py"           # Figs 7, 8
"${PY}" "${REPO}/08_external_validation/scripts/06_make_sge_figure.py"    # Fig. 9
echo "ALL DONE"
