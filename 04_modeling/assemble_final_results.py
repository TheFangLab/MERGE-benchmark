"""
Record both senses of "strongest single feature" in each task's meta.json.

`pipeline.py` writes `{task}_meta.json` with the strongest feature of the
ensemble's own candidate pool (AlphaGenome excluded). `singlefeature_benchmark.py`
then rewrites `{task}_singlefeature.csv` over the full benchmark panel, AlphaGenome
included. This step reads that table and stores

  best_single_feature                  the strongest feature of any kind
                                       (AlphaGenome on splicing and noncoding)
  best_single_nonalphagenome_feature   the strongest candidate input of MERGE,
                                       the reference for the Table 3 comparisons

Usage: python assemble_final_results.py
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS, TASKS


def is_ag(name):
    return str(name).startswith("alphagenome")


def main():
    for t in TASKS:
        path = Path(RESULTS) / f"{t}_meta.json"
        meta = json.loads(path.read_text())
        sf = pd.read_csv(Path(RESULTS) / f"{t}_singlefeature.csv").sort_values(
            "auroc_mean", ascending=False)
        overall = sf.iloc[0]
        nonag = sf[~sf["feature"].map(is_ag)].iloc[0]
        meta.update({
            "model": "MERGE without AlphaGenome",
            "candidate_pool_for_ensemble": "all features except alphagenome_*",
            "candidate_pool_for_single_feature_benchmark": "all features, AlphaGenome "
                "included and scored by a label-free percentile without fitting a model",
            "best_single_feature": str(overall["feature"]),
            "best_single_auroc": float(overall["auroc_mean"]),
            "best_single_auprc": float(overall["auprc_mean"]),
            "best_single_nonalphagenome_feature": str(nonag["feature"]),
            "best_single_nonalphagenome_auroc": float(nonag["auroc_mean"]),
            "best_single_nonalphagenome_auprc": float(nonag["auprc_mean"]),
            "alphagenome_average_offered": t in ("splicing", "noncoding"),
            "ensemble_excludes": "alphagenome_*",
        })
        path.write_text(json.dumps(meta, indent=2))
        print(f"[{t}] MERGE {meta['best_strategy']} {meta['merge_auroc']:.4f}/"
              f"{meta['merge_auprc']:.4f}; strongest feature {meta['best_single_feature']} "
              f"{meta['best_single_auroc']:.4f}; strongest candidate input "
              f"{meta['best_single_nonalphagenome_feature']} "
              f"{meta['best_single_nonalphagenome_auroc']:.4f}")


if __name__ == "__main__":
    main()
