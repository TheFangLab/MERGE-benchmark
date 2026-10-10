"""Shared configuration for the MERGE benchmark.

Every path defaults to a directory of this repository, so the scripts run from a
fresh clone without editing. Set the environment variables to write somewhere
else (for example, to regenerate the results next to the committed ones and
compare).
"""
import os
from pathlib import Path

RANDOM_STATE = 42

REPO = Path(__file__).resolve().parent.parent
DATA = os.environ.get("MERGE_DATA", str(REPO / "03_datasets"))
RESULTS = os.environ.get("MERGE_RESULTS", str(REPO / "05_results"))
BALANCED = os.environ.get("MERGE_BALANCED", os.path.join(RESULTS, "balanced_1to1"))
AGAVG = os.environ.get("MERGE_AGAVG", str(REPO / "05_results_alphagenome_average"))
FIGURES = os.environ.get("MERGE_FIGURES", str(REPO / "06_figures"))

ID_COLS = ["chr", "pos", "ref", "alt", "label"]

# Chromosome-held-out 3-fold cross-validation. Values are the held-out chromosomes.
FOLDS = {
    1: {"3", "6", "9", "12", "16", "22", "X", "Y"},
    2: {"2", "5", "11", "14", "17", "18", "19", "20", "21"},
    3: {"1", "4", "7", "8", "10", "13", "15"},
}

TASKS = {
    "coding":    "coding_features.csv",
    "splicing":  "splicing_features.csv",
    "noncoding": "noncoding_features.csv",
    "nonsnv":    "non_snv_features.csv",
}

# Features need at least 95% coverage and more than one distinct value.
MIN_COVERAGE = 0.95

# Column-name prefixes removed from the candidate pool before any selection runs.
# MERGE is never trained on AlphaGenome output: the AlphaGenome terms ask users not
# to train machine-learning models on it. AlphaGenome is still evaluated, as an
# independent comparator, by singlefeature_benchmark.py, which scores it by a
# label-free percentile without fitting anything.
ENSEMBLE_EXCLUDE_PREFIXES = ["alphagenome"]
