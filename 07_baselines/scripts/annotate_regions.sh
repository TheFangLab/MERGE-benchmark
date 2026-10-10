#!/usr/bin/env bash
# Region class for each variant, via ANNOVAR gene annotation.
#
#   bash annotate_regions.sh <sites.csv> <annovar-dir> <out-prefix>
#
# The point is stratified reporting. A non-coding set pooled across region
# classes inflates AUROC for every method, because the region composition
# differs between the positives and the negatives rather than because any
# method is discriminating well (Grosser et al., bioRxiv 2025.09.05.674459).
# With 1.5 % positives this dataset is exactly that shape, so the comparison is
# reported per region class as well as pooled.
set -euo pipefail
SITES="$1"; ANNOVAR="$2"; OUT="$3"

# ANNOVAR wants: chr start end ref alt. For an SNV start == end.
tail -n +2 "$SITES" | awk -F, 'BEGIN{OFS="\t"}
  {chr=$1; sub(/^chr/,"",chr);
   end=$2 + length($3) - 1;
   print "chr"chr, $2, end, $3, $4}' > "$OUT.avinput"
echo "$(wc -l < "$OUT.avinput") variants -> $OUT.avinput"

perl "$ANNOVAR/annotate_variation.pl" -geneanno -dbtype refGene \
     -buildver hg38 "$OUT.avinput" "$ANNOVAR/humandb/" \
     -out "$OUT" 2>&1 | tail -5

cut -f1 "$OUT.variant_function" | sort | uniq -c | sort -rn
