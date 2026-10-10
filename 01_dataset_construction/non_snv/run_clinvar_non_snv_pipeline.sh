#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
VCF="${VCF:-${REPO}/00_raw_data/clinvar/clinvar_20251221.vcf.gz}"
BASE_OUTDIR="${BASE_OUTDIR:-./work}"
WORKDIR="${WORKDIR:-${BASE_OUTDIR}/non_snv_pipeline}"
SCRIPT="${SCRIPT:-${HERE}/build_clinvar_non_snv_dataset.py}"
MAX_LEN="${MAX_LEN:-50}"
# SNV datasets to de-duplicate against, by (chr, pos, ref, alt)
DEDUP_CSVS="${DEDUP_CSVS:-${REPO}/03_datasets/coding_features.csv ${REPO}/03_datasets/noncoding_features.csv ${REPO}/03_datasets/splicing_features.csv}"
DEDUP_ARGS=()
for f in ${DEDUP_CSVS}; do DEDUP_ARGS+=(--dedup-csv "$f"); done

# Set these before running the ANNOVAR step, for example:
#   export ANNOVAR_DIR=/path/to/annovar
#   export ANNOVAR_DB=/path/to/annovar/humandb
ANNOVAR_DIR="${ANNOVAR_DIR:-}"
ANNOVAR_DB="${ANNOVAR_DB:-}"
BUILDVER="${BUILDVER:-hg38}"

mkdir -p "${WORKDIR}" "${BASE_OUTDIR}"

RAW_TSV="${WORKDIR}/clinvar_20251221.non_snv.raw.tsv"
PRE_CSV="${WORKDIR}/clinvar_20251221.non_snv.pre_annovar.csv"
AVINPUT="${WORKDIR}/clinvar_20251221.non_snv.avinput"
ANN_PREFIX="${WORKDIR}/clinvar_20251221.non_snv"
FINAL_CSV="${BASE_OUTDIR}/clinvar_20251221.non_snv.csv"
SPLICE_ISOLATED_CSV="${BASE_OUTDIR}/clinvar_20251221.non_snv.splice_boundary_isolated.csv"

echo "[1/5] Checking VCF readability: ${VCF}"
bcftools view -h "${VCF}" >/dev/null

echo "[2/5] Extracting raw Non-SNV candidates with bcftools"
bcftools view \
  -i 'strlen(REF)>1 || strlen(ALT)>1 || INFO/CLNVC="Insertion" || INFO/CLNVC="Deletion" || INFO/CLNVC="MNV"' \
  "${VCF}" \
| bcftools query \
  -f '%CHROM\t%POS\t%ID\t%REF\t%ALT\t%INFO/ALLELEID\t%INFO/CLNSIG\t%INFO/CLNREVSTAT\t%INFO/CLNVC\t%INFO/CLNVCSO\t%INFO/GENEINFO\t%INFO/MC\t%INFO/CLNDN\n' \
| awk 'BEGIN{OFS="\t"; print "CHROM","POS","ID","REF","ALT","ALLELEID","CLNSIG","CLNREVSTAT","CLNVC","CLNVCSO","GENEINFO","MC","CLNDN"} {print}' \
> "${RAW_TSV}"

echo "[3/5] Cleaning, labeling, high-confidence filtering, and writing ANNOVAR input"
python3 "${SCRIPT}" pre \
  --input-tsv "${RAW_TSV}" \
  --output-csv "${PRE_CSV}" \
  --output-avinput "${AVINPUT}" \
  --max-len "${MAX_LEN}"

echo "[4/5] Running ANNOVAR gene annotation with 2bp splice boundary"
if [[ -z "${ANNOVAR_DIR}" || -z "${ANNOVAR_DB}" ]]; then
  echo "ANNOVAR_DIR and ANNOVAR_DB are not set. Pre-ANNOVAR outputs are ready:"
  echo "  ${PRE_CSV}"
  echo "  ${AVINPUT}"
  echo "Run ANNOVAR manually, then run the finalize command printed below."
  echo
  echo "Example ANNOVAR command:"
  echo "  perl /path/to/annovar/annotate_variation.pl -geneanno -buildver ${BUILDVER} -dbtype refGene -splicing_threshold 2 ${AVINPUT} /path/to/annovar/humandb -out ${ANN_PREFIX}"
  echo
  echo "Finalize command:"
  echo "  python3 ${SCRIPT} finalize --pre-csv ${PRE_CSV} --annovar-variant-function ${ANN_PREFIX}.variant_function --output-csv ${FINAL_CSV} --isolated-splice-csv ${SPLICE_ISOLATED_CSV} ${DEDUP_ARGS[*]}"
  exit 0
fi

perl "${ANNOVAR_DIR}/annotate_variation.pl" \
  -geneanno \
  -buildver "${BUILDVER}" \
  -dbtype refGene \
  -splicing_threshold 2 \
  "${AVINPUT}" \
  "${ANNOVAR_DB}" \
  -out "${ANN_PREFIX}"

echo "[5/5] Applying splice-priority isolation and de-duplication"
python3 "${SCRIPT}" finalize \
  --pre-csv "${PRE_CSV}" \
  --annovar-variant-function "${ANN_PREFIX}.variant_function" \
  --output-csv "${FINAL_CSV}" \
  --isolated-splice-csv "${SPLICE_ISOLATED_CSV}" \
  "${DEDUP_ARGS[@]}"

echo "Done: ${FINAL_CSV}"
