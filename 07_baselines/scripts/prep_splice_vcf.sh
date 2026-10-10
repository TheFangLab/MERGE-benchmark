#!/usr/bin/env bash
# Adapt a benchmark VCF for SpliceAI and Pangolin.
#
#   bash prep_splice_vcf.sh <in.vcf> <hg38.fa.fai> <out.vcf>
#
# Two changes are needed that CADD did not care about:
#   * contig names must match the reference FASTA, which is UCSC-style ("chr1")
#     whereas the benchmark tables and the CADD service use Ensembl-style ("1");
#   * the header must declare every contig, because SpliceAI writes its output
#     through pysam, and pysam refuses to write a record whose contig the header
#     does not know ("Can't write record: Invalid argument").
set -euo pipefail
IN="$1"; FAI="$2"; OUT="$3"
{
  echo '##fileformat=VCFv4.2'
  awk '{printf "##contig=<ID=%s,length=%s>\n", $1, $2}' "$FAI"
  printf '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n'
  grep -v '^#' "$IN" | awk 'BEGIN{OFS="\t"} {if ($1 !~ /^chr/) $1="chr"$1; print}'
} > "$OUT"
echo "$(grep -vc '^#' "$OUT") variants -> $OUT"
