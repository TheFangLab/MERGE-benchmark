#!/usr/bin/env bash
# Step 3 — common-variant negatives for the splicing task, from gnomAD v4.1.
#
# Provenance: this is the working note "gnomAD阴性数据补充.txt", turned into a script.
#
# Rationale: a variant that sits in a splice region and still reaches a population allele
# frequency above 0.001 is, by that frequency alone, unlikely to abolish splicing. These
# are frequency-based proxy controls, not experimentally confirmed normal-splicing
# variants, and the manuscript says so.
#
# Pipeline:
#   per-chromosome gnomAD genome sites VCF
#     -> keep SNVs with AF > 0.001
#     -> ANNOVAR avinput
#     -> concatenate all chromosomes
#     -> ANNOVAR refGene annotation (default splicing threshold: within 2 bp of an
#        exon-intron boundary)
#     -> step 5 keeps the rows whose Func.refGene mentions splicing
#
# Reference counts from the original run: 27,718,937 variants pass AF > 0.001 genome-wide;
# 1,759 of them fall in a splice region under ANNOVAR's default 2 bp rule.
#
# Usage:
#   export ANNOVAR_DIR=/path/to/annovar ANNOVAR_DB=/path/to/annovar/humandb
#   GNOMAD_DIR=../../00_raw_data/gnomad WORKDIR=./work bash 04_gnomad_splice_negatives.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GNOMAD_DIR="${GNOMAD_DIR:-${HERE}/../../00_raw_data/gnomad}"
WORKDIR="${WORKDIR:-./work}"
BUILDVER="${BUILDVER:-hg38}"
AF_THRESHOLD="${AF_THRESHOLD:-0.001}"
CHROMS="${CHROMS:-$(seq 1 22) X Y}"
ANNOVAR_DIR="${ANNOVAR_DIR:-}"
ANNOVAR_DB="${ANNOVAR_DB:-}"

MERGED_AVINPUT="${WORKDIR}/gnomad_all_merged.avinput"
OUT_PREFIX="${OUT_PREFIX:-${WORKDIR}/gnomad_result}"

mkdir -p "${WORKDIR}"

echo "[1/2] Filtering gnomAD (SNVs, AF > ${AF_THRESHOLD}) into an ANNOVAR avinput"
if [[ -s "${MERGED_AVINPUT}" ]]; then
  echo "      exists, skipping: ${MERGED_AVINPUT}"
else
  : > "${MERGED_AVINPUT}.part"
  for c in ${CHROMS}; do
    vcf="${GNOMAD_DIR}/gnomad.genomes.v4.1.sites.chr${c}.vcf.bgz"
    if [[ ! -s "${vcf}" ]]; then
      echo "      missing, skipping: ${vcf}" >&2
      continue
    fi
    echo "      chr${c}"
    # avinput layout: chr start end ref alt [passthrough: AF]
    bcftools view -v snps -i "INFO/AF > ${AF_THRESHOLD}" "${vcf}" \
    | bcftools query -f '%CHROM\t%POS\t%POS\t%REF\t%ALT\t%INFO/AF\n' \
    >> "${MERGED_AVINPUT}.part"
  done
  mv "${MERGED_AVINPUT}.part" "${MERGED_AVINPUT}"
fi
echo "      $(wc -l < "${MERGED_AVINPUT}") variants"

echo "[2/2] ANNOVAR refGene annotation"
if [[ -z "${ANNOVAR_DIR}" || -z "${ANNOVAR_DB}" ]]; then
  cat <<MSG
ANNOVAR_DIR / ANNOVAR_DB are not set. The avinput is ready:
  ${MERGED_AVINPUT}

Run the annotation by hand with:
  perl \${ANNOVAR_DIR}/table_annovar.pl ${MERGED_AVINPUT} \${ANNOVAR_DB} \\
    -buildver ${BUILDVER} \\
    -out ${OUT_PREFIX} \\
    -remove \\
    -protocol refGene \\
    -operation g \\
    -nastring . \\
    -csvout
MSG
  exit 0
fi

perl "${ANNOVAR_DIR}/table_annovar.pl" "${MERGED_AVINPUT}" "${ANNOVAR_DB}" \
  -buildver "${BUILDVER}" \
  -out "${OUT_PREFIX}" \
  -remove \
  -protocol refGene \
  -operation g \
  -nastring . \
  -csvout

echo "Done: ${OUT_PREFIX}.${BUILDVER}_multianno.csv"
echo "Step 5 selects the splice-region rows from it."
