# MERGE-benchmark

Data and code for *Benchmarking genomic foundation and sequence-to-function models for
variant pathogenicity prediction* (Chen and Fang).

DNA language, evolutionary, protein and sequence-to-function models are benchmarked on
four labelled variant classes — coding missense SNVs, splicing SNVs, noncoding SNVs and
short non-SNVs — under fixed chromosome-held-out folds. MERGE (Model Ensemble for Ranking
Genomic Effects) integrates fixed model-derived features to measure their
complementarity. AlphaGenome is evaluated independently and through a fixed MERGE +
AlphaGenome average; its outputs are never used for supervised feature selection or
model fitting.

## Manuscript items

| Item | Files | Code |
|---|---|---|
| Tables 1–10 | `05_results/tables/` (`TABLES.md`, one CSV per table) | `04_modeling/manuscript_tables.py` |
| Figures 2–9 | `06_figures/` | see `06_figures/README.md` |
| Datasets (Table 10) | `03_datasets/` | `01_dataset_construction/` |
| Feature extraction | columns of `03_datasets/` | `02_feature_extraction/` |
| Individual features (Table 1) | `05_results/{task}_singlefeature*.csv` | `04_modeling/singlefeature_benchmark.py` |
| MERGE (Tables 2, 4, 6; Figs 4–6) | `05_results/{task}_merge_*`, `05_results/balanced_1to1/` | `04_modeling/pipeline.py` |
| MERGE + AlphaGenome average (Table 2) | `05_results_alphagenome_average/` | `04_modeling/alphagenome_average.py` |
| Paired tests, calibration (Tables 3, 5) | `05_results/stats/`, `calibration/`, `stats_cluster/` | `04_modeling/stats_tests.py`, `calibration_and_cluster_bootstrap.py` |
| Established predictors (Tables 7–8) | `07_baselines/results/` | `07_baselines/scripts/` |
| BRCA1 saturation genome editing (Table 9) | `08_external_validation/results/` | `08_external_validation/scripts/` |
| Deployed models | `05_results/trained_models/` | `04_modeling/export_models.py` |

## Layout

| Directory | Contents |
|---|---|
| `00_raw_data/` | Sources, versions, download script and checksums (large files are not stored) |
| `01_dataset_construction/` | ClinVar, SpliceVarDB and gnomAD processing into the four labelled datasets |
| `02_feature_extraction/` | Scoring scripts for every model and the per-task settings |
| `03_datasets/` | The four feature tables: the inputs of all analyses |
| `04_modeling/` | Benchmark, supervised integration, statistics, tables and figures |
| `05_results/` | Out-of-fold predictions, metrics, statistics, manuscript tables and deployed models |
| `05_results_alphagenome_average/` | The fixed MERGE + AlphaGenome average and its reference table |
| `06_figures/` | Figures (PNG and 300-dpi TIFF) |
| `07_baselines/` | Comparison with established predictors |
| `08_external_validation/` | BRCA1 saturation genome editing validation and the BRCA1-excluded retraining |
| `environment/` | Software versions and hardware |

| Task | Variants | Positive / negative | Benchmark features | MERGE candidate features |
|---|---:|---:|---:|---:|
| Coding missense SNVs | 4,289 | 2,912 / 1,377 | 25 | 19 |
| Splicing SNVs | 35,015 | 22,247 / 12,768 | 18 | 17 |
| Noncoding SNVs | 86,829 | 1,340 / 85,489 | 23 | 17 |
| Non-SNVs | 36,862 | 30,022 / 6,840 | 22 | 16 |

## Reproducing the analyses

The feature tables are committed, so every result, table and figure can be regenerated
on a CPU without downloads:

```bash
conda create -n merge_ml -c conda-forge python=3.11
conda activate merge_ml
pip install -r environment/requirements-modeling.txt
bash 04_modeling/run_all.sh          # about two hours on 16 cores; results are rewritten in place
git diff --stat                      # what changed
```

Seed 42 is used throughout. Multi-threaded tree learners depend on the number of CPU
cores, so on hardware other than the original 72-core node a few fold-level metrics can
differ in the fourth decimal (see `environment/README.md`); the committed files are the
results reported in the manuscript.

Rebuilding the datasets needs the resources in `00_raw_data/` (ClinVar, gnomAD,
SpliceVarDB, hg38, ANNOVAR), and recomputing the feature columns needs GPUs, the model
checkpoints and an AlphaGenome API key (`02_feature_extraction/`). The comparator scores
are committed in `07_baselines/scores/`; extracting them again needs dbNSFP 5.3a, the
noncoding score tracks, SpliceAI, Pangolin and the CADD web service.

## Using MERGE

* **Web server**: <https://merge.fanglab.cn> — single variants or VCF upload, hg38 or hg19.
* **Command line**: `pip install merge-cli` (command `merge`), documented at
  <https://github.com/TheFangLab/MERGE>; it also runs as a Model Context Protocol server.
* **Precomputed scores** for 107,922 coding and 1,920,006 splice-region SNVs: Zenodo
  (DOI to be added).

Both tools serve the models in `05_results/trained_models/` and, for splicing and
noncoding variants with an AlphaGenome score, the fixed average.

## Licence

Code and models: MIT. Derived tables and figures: CC BY 4.0. The AlphaGenome reference
table: CC BY-NC 4.0. Third-party scores extracted for the benchmark variants keep their
sources' terms. Details in [LICENSE](LICENSE).

The licences cover what this work produced, not the resources it uses. In particular,
the AlphaGenome API is offered for non-commercial use and its terms ask that outputs not
be used to train other models; dbNSFP (the source of the AlphaMissense, ESM-1b and most
coding comparator scores), Nucleotide Transformer, CADD and ANNOVAR restrict commercial
use. Users are responsible for complying with each provider's terms.

## Citation

Please cite the manuscript and this archive (`CITATION.cff`; Zenodo DOI to be added on
release).
