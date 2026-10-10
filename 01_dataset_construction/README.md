# 01_dataset_construction — raw resources → the four labelled datasets

Two subdirectories, one per provenance track.

```
coding_noncoding_splicing/        the ClinVar SNV tasks, run 2026-05/06
  00_annotate_clinvar_snv.sh        SNV-only ClinVar + ANNOVAR refGene
  01_build_coding_dataset.py        coding missense, from the VCF (MC = missense)
  02_build_noncoding_dataset.py     non-coding, from the ANNOVAR table
  03_build_clinvar_splicing.py      ClinVar splicing, from the VCF (MC = splice terms)
  04_gnomad_splice_negatives.sh     gnomAD AF > 0.001 splice-region negatives
  05_clean_splicevardb.py           SpliceVarDB strict exonic / intronic subsets
  06_assemble_splicing_dataset.py   the three splicing sources -> one table (35,774 rows)
  07_strict_deduplicate_splicing.py drop every duplicated variant key -> 35,015 rows
  splicing_strict_dedup_audit.csv   the counts of step 07
  run_all.sh, test_smoke.py
  README.md                         provenance, run order, filters, verification

non_snv/                          the non-SNV task, run 2026-06-15
  build_clinvar_non_snv_dataset.py
  run_clinvar_non_snv_pipeline.sh
```

Read `coding_noncoding_splicing/README.md` before using that directory: it records which
files are the original scripts and how each filter is applied.

## non_snv/

This is the script that produced `clinvar_20251221.non_snv.csv`, the basis of
`03_datasets/non_snv_features.csv`. Two stages:

1. `pre` — `bcftools` extract → keep records with `strlen(REF)>1 || strlen(ALT)>1` or a
   ClinVar `Insertion` / `Deletion` / `MNV` flag, drop anything where both alleles are a
   single nucleotide, require review status ≥ 2 stars, drop conflicting interpretations,
   drop alleles longer than 50 bp, assign the binary label, and write an ANNOVAR
   `avinput`.
2. `finalize` — read the ANNOVAR `variant_function` output produced with
   `-splicing_threshold 2`, move anything annotated `splicing` to a separate
   splice-boundary file, and de-duplicate against the coding, non-coding and splicing
   datasets by `(chr, pos, ref, alt)`.

```bash
export ANNOVAR_DIR=/path/to/annovar ANNOVAR_DB=/path/to/annovar/humandb
VCF=../../00_raw_data/clinvar/clinvar_20251221.vcf.gz \
BASE_OUTDIR=./out WORKDIR=./out/non_snv_pipeline \
bash non_snv/run_clinvar_non_snv_pipeline.sh
```

Without `ANNOVAR_DIR` / `ANNOVAR_DB` the script stops after stage 1 and prints the exact
ANNOVAR and finalize commands to run by hand.

## coding_noncoding_splicing/

```bash
export ANNOVAR_DIR=/path/to/annovar ANNOVAR_DB=/path/to/annovar/humandb
export SPLICEVARDB_RAW=/path/to/all_splicevardb_raw_data.csv
export GNOMAD_DIR=../../00_raw_data/gnomad          # optional, hundreds of GB
WORKDIR=./work bash coding_noncoding_splicing/run_all.sh
```

Each step also runs standalone; see that directory's README for the dependency graph and
for the expected row counts at every stage.

## What comes next

The datasets produced here carry only `chr, pos, ref, alt, label` plus provenance
columns. Feature columns are added by the scoring code in `02_feature_extraction/`.
