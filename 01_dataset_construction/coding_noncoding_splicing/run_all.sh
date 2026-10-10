#!/usr/bin/env bash
# Run the whole coding / non-coding / splicing construction in order.
#
#   export ANNOVAR_DIR=/path/to/annovar ANNOVAR_DB=/path/to/annovar/humandb
#   export SPLICEVARDB_RAW=/path/to/all_splicevardb_raw_data.csv
#   WORKDIR=./work bash run_all.sh
#
# Only step 2 needs the ANNOVAR annotation from step 0; the coding (01) and ClinVar
# splicing (03) extractions read the VCF directly. gnomAD (step 4) is skipped unless
# GNOMAD_DIR points at the downloaded genome sites VCFs; step 6 then assembles whatever
# splicing sources are available and reports what it used.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${WORKDIR:-./work}"
PYTHON="${PYTHON:-python3}"
BUILDVER="${BUILDVER:-hg38}"

VCF="${VCF:-${HERE}/../../00_raw_data/clinvar/clinvar_20251221.vcf.gz}"
GNOMAD_DIR="${GNOMAD_DIR:-}"
SPLICEVARDB_RAW="${SPLICEVARDB_RAW:-}"

mkdir -p "${WORKDIR}"
export WORKDIR VCF BUILDVER

ANNOTATED="${WORKDIR}/clinvar_annotated.${BUILDVER}_multianno.txt"

echo "=== Step 0: ClinVar SNVs -> ANNOVAR refGene ==="
bash "${HERE}/00_annotate_clinvar_snv.sh"

echo
echo "=== Step 1: coding missense dataset (straight from the VCF) ==="
"${PYTHON}" "${HERE}/01_build_coding_dataset.py" \
  --clinvar-vcf "${VCF}" \
  --output-csv "${WORKDIR}/coding_snv.csv"

echo
echo "=== Step 3: ClinVar splicing component (straight from the VCF, MC field) ==="
"${PYTHON}" "${HERE}/03_build_clinvar_splicing.py" \
  --clinvar-vcf "${VCF}" \
  --output-csv "${WORKDIR}/clinvar_splicing_snv_2star.csv"

if [[ -s "${ANNOTATED}" ]]; then
  echo
  echo "=== Step 2: non-coding dataset ==="
  "${PYTHON}" "${HERE}/02_build_noncoding_dataset.py" \
    --annotated "${ANNOTATED}" \
    --output-csv "${WORKDIR}/noncoding_snv.csv"
else
  echo
  echo "!! ${ANNOTATED} is missing — ANNOVAR did not run, so the non-coding dataset is skipped."
  echo "   Set ANNOVAR_DIR / ANNOVAR_DB and re-run."
fi

if [[ -n "${SPLICEVARDB_RAW}" ]]; then
  echo
  echo "=== Step 5: SpliceVarDB strict subsets ==="
  "${PYTHON}" "${HERE}/05_clean_splicevardb.py" \
    --input "${SPLICEVARDB_RAW}" \
    --output-exonic "${WORKDIR}/SpliceVarDB_Exonic_SNV_Strict.csv" \
    --output-intronic "${WORKDIR}/SpliceVarDB_Intronic_SNV_Strict.csv"
else
  echo
  echo "=== Step 5 skipped: set SPLICEVARDB_RAW to the SpliceVarDB export ==="
fi

GNOMAD_MULTIANNO="${WORKDIR}/gnomad_result.${BUILDVER}_multianno.csv"
if [[ -n "${GNOMAD_DIR}" ]]; then
  echo
  echo "=== Step 4: gnomAD common splice-region negatives ==="
  GNOMAD_DIR="${GNOMAD_DIR}" bash "${HERE}/04_gnomad_splice_negatives.sh"
else
  echo
  echo "=== Step 4 skipped: set GNOMAD_DIR to the gnomAD v4.1 genome sites directory ==="
fi

if [[ -s "${WORKDIR}/SpliceVarDB_Exonic_SNV_Strict.csv" ]]; then
  echo
  echo "=== Step 6: assemble the splicing dataset ==="
  ARGS=(--splicevardb "${WORKDIR}/SpliceVarDB_Exonic_SNV_Strict.csv"
        --splicevardb "${WORKDIR}/SpliceVarDB_Intronic_SNV_Strict.csv"
        --clinvar "${WORKDIR}/clinvar_splicing_snv_2star.csv"
        --output-csv "${WORKDIR}/splicing_snv.csv")
  if [[ -s "${GNOMAD_MULTIANNO}" ]]; then
    ARGS+=(--gnomad-multianno "${GNOMAD_MULTIANNO}")
  else
    echo "    (no gnomAD annotation found — assembling SpliceVarDB + ClinVar only,"
    echo "     which is 34,015 of the 35,774 assembled rows)"
  fi
  "${PYTHON}" "${HERE}/06_assemble_splicing_dataset.py" "${ARGS[@]}"

  echo
  echo "=== Step 7: strict deduplication (every duplicated variant removed) ==="
  "${PYTHON}" "${HERE}/07_strict_deduplicate_splicing.py" \
      "${WORKDIR}/splicing_snv.csv" \
      "${WORKDIR}/splicing_snv_strict_dedup.csv" \
      "${WORKDIR}/splicing_strict_dedup_audit.csv"
else
  echo
  echo "=== Step 6 skipped: needs the SpliceVarDB output ==="
fi

echo
echo "Done. Outputs in ${WORKDIR}"
