#!/usr/bin/env bash
# SpliceAI over a benchmark VCF, sharded across processes.
#
#   bash run_spliceai.sh <input.vcf> <out-prefix> <hg38.fa> [n-shards]
#
# SpliceAI ships its own GENCODE annotation table (-A grch38), so no external
# GTF is needed. One process per shard: the model is small and the run is
# dominated by per-variant work, so sharding scales close to linearly. Keep the
# shard count well below the core count -- this node also carries a long-running
# GPU job whose CPU-side feeding must not be starved.
# TensorFlow already parallelises inside one process, so the shards must be
# told to stay narrow -- otherwise N shards each open a machine-wide thread pool
# and spend their time contending rather than predicting. Two threads per shard
# times eight shards is about half this node, which leaves the long-running GPU
# job's CPU-side feeding alone.
set -euo pipefail
export OMP_NUM_THREADS=2 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1
export OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 OMP_WAIT_POLICY=passive
export TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES=""
IN="$1"; OUT="$2"; FASTA="$3"; N="${4:-8}"
# Prefer a copy of the reference staged in tmpfs. SpliceAI reads it by random
# access through pyfaidx, and on this cluster /home is NFS, which is what limits
# throughput rather than the model.
[[ -s /dev/shm/merge_ref/hg38.fa ]] && FASTA=/dev/shm/merge_ref/hg38.fa
echo "using FASTA=$FASTA"
# Shard directory is keyed to the output prefix. Deriving it from the output's
# directory alone let two runs share one workspace: the second run's tidy-up
# concatenated the first run's shard outputs into the second run's result file.
WORK="${OUT}_shards"
mkdir -p "$WORK"

grep '^#' "$IN" > "$WORK/header.txt"
grep -v '^#' "$IN" > "$WORK/body.txt"
total=$(wc -l < "$WORK/body.txt")
per=$(( (total + N - 1) / N ))
split -l "$per" -d -a 3 "$WORK/body.txt" "$WORK/shard."

pids=()
for s in "$WORK"/shard.*; do
  [[ "$s" == *.vcf ]] && continue
  cat "$WORK/header.txt" "$s" > "$s.vcf"
  ( spliceai -I "$s.vcf" -O "$s.out.vcf" -R "$FASTA" -A grch38 \
      > "$s.log" 2>&1 ) &
  pids+=($!)
done
echo "launched ${#pids[@]} shards over $total variants"
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
[[ $fail -gt 0 ]] && echo "WARNING: $fail shard(s) exited non-zero; check $WORK/*.log"

grep -h '^#' "$WORK"/shard.000.out.vcf > "$OUT.vcf" 2>/dev/null || true
grep -hv '^#' "$WORK"/shard.*.out.vcf >> "$OUT.vcf"
echo "SPLICEAI_DONE $(grep -vc '^#' "$OUT.vcf") rows -> $OUT.vcf"
