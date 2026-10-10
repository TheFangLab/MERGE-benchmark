# 08_external_validation — BRCA1 saturation genome editing

External validation of the frozen MERGE-Coding model on the BRCA1 saturation genome
editing (SGE) data of Findlay et al. (2018) (manuscript Table 9 and Figure 9). The SGE
labels are not used for feature selection, orientation, preprocessing or fitting.

## Cohort

From `source/GSE117159_Supplementary_Table_1.xlsx` (GEO GSE117159): missense SNVs
classified LOF or FUNC that were absent from ClinVar at the source time, lifted from
hg19 to hg38 with a verified hg38 reference base, and with no variant overlap with the
coding training table. The step-by-step counts are in `cohort/sge_cohort_flow.csv`.

| | n |
|---|---:|
| SGE missense cohort | 1,577 (289 LOF, 1,288 FUNC) |
| Scored by every predictor (common subset) | 1,421 (263 LOF, 1,158 FUNC) |

## Pipeline

| step | script | output |
|---|---|---|
| 1 | `scripts/01_build_sge_cohort.py --chain … --fasta … --clinvar …` | `cohort/` |
| 2 | feature extraction with the coding-task settings of `02_feature_extraction/` (GPU) | `model_outputs/*.csv.gz` |
| 3 | `scripts/02_extract_sge_dbnsfp.py --dbnsfp dbNSFP5.3a.zip` | `model_outputs/sge_dbnsfp_scores.csv` |
| 4 | `scripts/03_assemble_sge_features.py` | `features/sge_features.csv` |
| 5 | `scripts/04_evaluate_sge.py` | `results/sge_*` |
| 6 | `scripts/05_retrain_without_brca1.py` | `models/MERGE_coding_noBRCA1.pkl`, `results/*noBRCA1*`, `results/paired_with_vs_without_BRCA1.csv` |

Steps 4–6 run from the committed files in a few minutes on a CPU; steps 1–3 need the
external resources listed in `00_raw_data/` (hg19-to-hg38 chain, hg38 FASTA, ClinVar
VCF, dbNSFP 5.3a), which are not redistributed.

Notes on the inputs:

* `model_outputs/` holds the per-model outputs for the cohort, produced with the same
  checkpoints, scripts and parameters as the coding benchmark. GPN-MSA comes from the
  official precomputed table (`gpn_msa_brca1_chr17_43044295_43125364.tsv.gz`, the
  BRCA1 region, matched in `sge_GPN_MSA_score.csv`).
* AlphaGenome is reported per scorer by `02_feature_extraction/shared/alphagenome_non_snv.py`;
  step 4 reduces it to the six benchmark summaries over the 11 non-`*_active`,
  non-contact-map scorers (maximum, minimum, and contribution-weighted mean of the raw
  and quantile scores).
* `sge_dbnsfp_scores.csv` stores ESM-1b, SIFT4G and PROVEAN negated (larger = more
  damaging); step 4 restores the raw ESM-1b LLR used by the coding feature table.

## Evaluation

* **MERGE** is `05_results/trained_models/MERGE_noAlphaGenome_coding.pkl`, the deployed
  coding model, refitted on all 4,289 coding training variants with the
  benchmark-selected beam-search strategy and frozen before this cohort was scored.
* **Individual features** (25): AlphaGenome summaries use the direction stated in
  `04_modeling/singlefeature_benchmark.py`; every other feature takes the sign of its
  AUROC on the coding training table.
* **Established predictors** (17) from dbNSFP 5.3a. VEST4, MetaRNN, M-CAP, REVEL,
  BayesDel, ClinPred, VARITY and MutFormer were trained on clinical labels.
* Intervals and two-sided P values: 2,000 paired, class-stratified, variant-level
  bootstrap resamples on the common subset.

**Gene-level exclusion.** Step 6 removes the 212 coding training variants in the BRCA1
gene body (hg38 chr17:43,044,295–43,125,364), repeats feature selection, algorithm
choice and fitting on the remaining 4,077, and compares the result with the frozen
model on the same 1,421 variants.

## Results

| | AUROC (95% CI) | AUPRC (95% CI) |
|---|---|---|
| MERGE | 0.910 (0.890–0.929) | 0.716 (0.663–0.768) |
| AlphaMissense | 0.899 (0.877–0.919) | 0.697 (0.645–0.749) |
| VEST4 | 0.895 (0.871–0.915) | 0.706 (0.655–0.756) |
| MERGE retrained without BRCA1 | 0.910 | 0.714 |

MERGE minus AlphaMissense: ΔAUROC 0.011 (0.001–0.022), P = 0.027. Excluding BRCA1 from
training changes AUROC by −0.0004 (−0.0029 to 0.0022), P = 0.72. All 43 predictors are
in `results/sge_external_validation_metrics.csv`; every paired difference is in
`results/sge_paired_bootstrap_deltas.csv`.
