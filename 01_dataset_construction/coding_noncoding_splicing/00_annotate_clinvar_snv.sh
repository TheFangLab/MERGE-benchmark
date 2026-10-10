#!/usr/bin/env bash
# Step 0 — extract the SNV-only subset of ClinVar and annotate it with ANNOVAR refGene.
#
# Provenance: this is the working note "annovar注释脚本.txt", turned into a script.
# Its output (`${OUT_PREFIX}.hg38_multianno.txt`, called MERGE_clinvar_annotated_final.txt
# in the original working directory) is the input of step 02, which carves the non-coding
# and splicing partitions out of it.
#
# The coding dataset does NOT come from this file — step 01 reads the ClinVar VCF directly
# with pysam and uses ClinVar's own `MC` molecular-consequence field to select missense
# variants. That is how the released coding table was built.
#
# Usage:
#   export ANNOVAR_DIR=/path/to/annovar ANNOVAR_DB=/path/to/annovar/humandb
#   VCF=../../00_raw_data/clinvar/clinvar_20251221.vcf.gz WORKDIR=./work \
#     bash 00_annotate_clinvar_snv.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VCF="${VCF:-${HERE}/../../00_raw_data/clinvar/clinvar_20251221.vcf.gz}"
WORKDIR="${WORKDIR:-./work}"
BUILDVER="${BUILDVER:-hg38}"
ANNOVAR_DIR="${ANNOVAR_DIR:-}"
ANNOVAR_DB="${ANNOVAR_DB:-}"

SNV_VCF="${WORKDIR}/clinvar_snps_only.vcf.gz"
OUT_PREFIX="${OUT_PREFIX:-${WORKDIR}/clinvar_annotated}"

mkdir -p "${WORKDIR}"

echo "[1/3] Keeping SNVs only (drops indels and structural alleles)"
if [[ -s "${SNV_VCF}" ]]; then
  echo "      exists, skipping: ${SNV_VCF}"
else
  bcftools view -v snps "${VCF}" -O z -o "${SNV_VCF}"
fi

echo "[2/3] Indexing"
[[ -s "${SNV_VCF}.tbi" ]] || bcftools index -t "${SNV_VCF}"

echo "[3/3] ANNOVAR refGene gene-based annotation"
if [[ -z "${ANNOVAR_DIR}" || -z "${ANNOVAR_DB}" ]]; then
  cat <<MSG
ANNOVAR_DIR / ANNOVAR_DB are not set. The SNV-only VCF is ready:
  ${SNV_VCF}

Run the annotation by hand with:
  perl \${ANNOVAR_DIR}/table_annovar.pl ${SNV_VCF} \${ANNOVAR_DB} \\
    -buildver ${BUILDVER} \\
    -out ${OUT_PREFIX} \\
    -remove \\
    -protocol refGene \\
    -operation g \\
    -nastring . \\
    -vcfinput

(The refGene database itself comes from:
  perl \${ANNOVAR_DIR}/annotate_variation.pl -buildver ${BUILDVER} -downdb -webfrom annovar refGene \${ANNOVAR_DB}/)
MSG
  exit 0
fi

perl "${ANNOVAR_DIR}/table_annovar.pl" "${SNV_VCF}" "${ANNOVAR_DB}" \
  -buildver "${BUILDVER}" \
  -out "${OUT_PREFIX}" \
  -remove \
  -protocol refGene \
  -operation g \
  -nastring . \
  -vcfinput

echo "Done: ${OUT_PREFIX}.${BUILDVER}_multianno.txt"
