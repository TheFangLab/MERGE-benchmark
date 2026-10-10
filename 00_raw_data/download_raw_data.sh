#!/usr/bin/env bash
# Fetch the public source data used by the MERGE benchmark.
#
#   bash download_raw_data.sh              # everything except gnomAD
#   bash download_raw_data.sh gnomad       # gnomAD v4.1 genome sites (hundreds of GB)
#   bash download_raw_data.sh clinvar hg38 # pick individual targets
#
# Targets: clinvar hg38 gencode dbnsfp gpnmsa splicevardb annovar gnomad
#
# Everything lands under the directory this script lives in. Disk needed for the
# default set: about 12 GB (hg38 dominates). gnomAD is opt-in because the genome
# sites VCFs are several hundred GB in total.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

CLINVAR_RELEASE="${CLINVAR_RELEASE:-clinvar_20251221}"
GENCODE_RELEASE="${GENCODE_RELEASE:-44}"

have() { command -v "$1" >/dev/null 2>&1; }
fetch() {  # fetch <url> <destination>
  local url="$1" dest="$2"
  if [[ -s "${dest}" ]]; then
    echo "  exists, skipping: ${dest}"
    return 0
  fi
  mkdir -p "$(dirname "${dest}")"
  echo "  -> ${url}"
  if have curl; then
    curl -fL --retry 5 --retry-delay 10 -C - -o "${dest}.part" "${url}"
  elif have wget; then
    wget -c -O "${dest}.part" "${url}"
  else
    echo "Neither curl nor wget is available." >&2
    return 1
  fi
  mv "${dest}.part" "${dest}"
}

# ------------------------------------------------------------------------------------
get_clinvar() {
  echo "[clinvar] ${CLINVAR_RELEASE} (GRCh38)"
  local base="https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38"
  # Recent releases sit in weekly/; older ones are moved to archive_2.0/<year>/.
  local year="${CLINVAR_RELEASE:8:4}"
  for prefix in "${base}/weekly" "${base}/archive_2.0/${year}"; do
    if fetch "${prefix}/${CLINVAR_RELEASE}.vcf.gz" "clinvar/${CLINVAR_RELEASE}.vcf.gz" 2>/dev/null; then
      fetch "${prefix}/${CLINVAR_RELEASE}.vcf.gz.tbi" "clinvar/${CLINVAR_RELEASE}.vcf.gz.tbi" || true
      return 0
    fi
  done
  echo "  !! ${CLINVAR_RELEASE} was not found under weekly/ or archive_2.0/${year}/." >&2
  echo "     Browse ${base}/ and set CLINVAR_RELEASE to an available release." >&2
  return 1
}

get_hg38() {
  echo "[hg38] UCSC reference genome"
  fetch "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz" "genome/hg38.fa.gz"
  fetch "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/md5sum.txt"  "genome/ucsc_md5sum.txt"
  if [[ ! -s genome/hg38.fa ]]; then
    echo "  decompressing (3.2 GB uncompressed)"
    gunzip -c genome/hg38.fa.gz > genome/hg38.fa
  fi
  if have samtools; then
    [[ -s genome/hg38.fa.fai ]] || samtools faidx genome/hg38.fa
  else
    echo "  samtools not found; create genome/hg38.fa.fai before running the scoring code."
  fi
}

get_gencode() {
  echo "[gencode] release ${GENCODE_RELEASE}"
  fetch "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_${GENCODE_RELEASE}/gencode.v${GENCODE_RELEASE}.annotation.gtf.gz" \
        "gencode/gencode.v${GENCODE_RELEASE}.annotation.gtf.gz"
}

get_dbnsfp() {
  echo "[dbnsfp] manual download required"
  cat <<'MSG'
  AlphaMissense and ESM-1b scores, and the coding comparators of 07_baselines/, were
  taken from dbNSFP 5.3a (academic release; commercial use needs a separate licence).
  Download dbNSFP5.3a from

      https://www.dbnsfp.org/releases/

  and pass the archive to the extraction scripts (07_baselines/scripts/extract_dbnsfp.py,
  08_external_validation/scripts/02_extract_sge_dbnsfp.py).
MSG
}

get_gpnmsa() {
  echo "[gpn-msa] hg38 genome-wide scores"
  if have huggingface-cli; then
    huggingface-cli download songlab/gpn-msa-hg38-scores --repo-type dataset --local-dir gpn_msa
  else
    echo "  huggingface-cli not found. Install it (pip install -U 'huggingface_hub[cli]') and run:"
    echo "    huggingface-cli download songlab/gpn-msa-hg38-scores --repo-type dataset --local-dir ${HERE}/gpn_msa"
  fi
}

get_splicevardb() {
  echo "[splicevardb] manual download required"
  cat <<'MSG'
  SpliceVarDB has no stable direct download URL. Open

      https://splicevardb.org/

  export the full variant table and save it as

      splicevardb/all_splicevardb_raw_data.csv

  Required columns:
      chr, pos_hg38, ref, alt, classification, location

  01_dataset_construction/coding_noncoding_splicing/05_clean_splicevardb.py splits it
  into the strict exonic and intronic subsets.
MSG
  mkdir -p splicevardb
}

get_annovar() {
  echo "[annovar] manual download required"
  cat <<'MSG'
  ANNOVAR needs a free academic registration:

      https://www.openbioinformatics.org/annovar/annovar_download_form.php

  After unpacking, install the hg38 refGene database:

      perl annovar/annotate_variation.pl -downdb -buildver hg38 -webfrom annovar refGene annovar/humandb/

  Then export, before running 01_dataset_construction:

      export ANNOVAR_DIR=/path/to/annovar
      export ANNOVAR_DB=/path/to/annovar/humandb
MSG
}

get_gnomad() {
  echo "[gnomad] v4.1 genome sites — hundreds of GB, opt-in"
  local base="https://storage.googleapis.com/gcp-public-data--gnomad/release/4.1/vcf/genomes"
  mkdir -p gnomad
  for c in $(seq 1 22) X Y; do
    fetch "${base}/gnomad.genomes.v4.1.sites.chr${c}.vcf.bgz"     "gnomad/gnomad.genomes.v4.1.sites.chr${c}.vcf.bgz"
    fetch "${base}/gnomad.genomes.v4.1.sites.chr${c}.vcf.bgz.tbi" "gnomad/gnomad.genomes.v4.1.sites.chr${c}.vcf.bgz.tbi"
  done
  cat <<'MSG'

  The splicing dataset uses only common variants (AF > 0.001) that ANNOVAR places
  within 2 bp of an exon-intron boundary. Both steps are done by

      01_dataset_construction/coding_noncoding_splicing/04_gnomad_splice_negatives.sh

  which filters every per-chromosome VCF, concatenates the result into one ANNOVAR
  avinput and annotates it with refGene. No interval file is needed: ANNOVAR's own
  default splicing threshold of 2 bp does the boundary selection.
MSG
}

TARGETS=("$@")
if [[ ${#TARGETS[@]} -eq 0 ]]; then
  TARGETS=(clinvar hg38 gencode dbnsfp gpnmsa splicevardb annovar)
fi

for t in "${TARGETS[@]}"; do
  case "$t" in
    clinvar)       get_clinvar ;;
    hg38)          get_hg38 ;;
    gencode)       get_gencode ;;
    dbnsfp)        get_dbnsfp ;;
    gpnmsa)        get_gpnmsa ;;
    splicevardb)   get_splicevardb ;;
    annovar)       get_annovar ;;
    gnomad)        get_gnomad ;;
    *) echo "Unknown target: $t" >&2; exit 2 ;;
  esac
done

echo
echo "Done. Verify with: md5sum -c checksums.md5"
