#!/usr/bin/env bash
# Pangolin over the splicing and non-SNV datasets, sharded across GPUs.
#
#   bash run_pangolin.sh [base-dir] [gpu-list] [shards-per-gpu]
#
# Two things are worth knowing before changing this.
#
# CSV input, not VCF: Pangolin's VCF writer calls vcf.parser._Info with the
# original PyVCF signature, and PyVCF3 -- the only fork that installs on
# Python 3.11 -- added a `type_code` field to that namedtuple, so the VCF path
# dies with "Info.__new__() missing 1 required positional argument". The CSV
# path yields identical scores without patching an installed package.
#
# Sharded, because Pangolin scores roughly 137 variants a minute in one process
# regardless of the GPU -- the run is dominated by per-variant overhead, not by
# the model -- which would put 72,000 variants at nine hours.
# PyTorch opens an intra-op thread pool the width of the machine even when the
# model runs on the GPU, so each shard took about five cores of preprocessing.
# Eight unconstrained shards drove a 48-core node to a load average of 170 and
# starved the SpliceAI run sharing it. Two threads per shard is enough to keep
# the GPUs fed.
set -uo pipefail
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export OMP_WAIT_POLICY=passive
B="${1:-$HOME/merge_baselines}"
GPUS="${2:-3,5,6,7}"
PER_GPU="${3:-2}"
P="$HOME/miniconda3/envs/splicebase/bin"
# Prefer copies staged in tmpfs. The annotation database is SQLite and the
# reference is read by random access; on this cluster /home is NFS, and serving
# either over NFS is what limits throughput, not the GPU. Eight shards on NFS
# managed only three times the speed of one; the files are 3.6 GB together and
# /dev/shm has 252 GB.
SHM=/dev/shm/merge_ref
DB="$B/data/gencode.v38.annotation.db"
FASTA="$HOME/genomes/hg38.fa"
[[ -s "$SHM/gencode.v38.annotation.db" ]] && DB="$SHM/gencode.v38.annotation.db"
[[ -s "$SHM/hg38.fa" ]] && FASTA="$SHM/hg38.fa"
echo "using DB=$DB FASTA=$FASTA"
cd "$B"
mkdir -p splice_work/pangolin_shards scores logs
IFS=',' read -ra GPU_ARR <<< "$GPUS"
N=$(( ${#GPU_ARR[@]} * PER_GPU ))

for task in splicing nonsnv; do
  W="splice_work/pangolin_shards/$task"
  rm -rf "$W"; mkdir -p "$W"
  grep -v '^#' "vcf/$task.chr.vcf" \
    | awk -F'\t' 'BEGIN{OFS=","}{print $1,$2,$4,$5}' > "$W/body.csv"
  total=$(wc -l < "$W/body.csv")
  per=$(( (total + N - 1) / N ))
  split -l "$per" -d -a 3 "$W/body.csv" "$W/part."
  echo "=== pangolin $task: $total variants, $N shards $(date +%H:%M:%S) ==="

  k=0; pids=()
  for f in "$W"/part.*; do
    [[ "$f" == *.csv ]] && continue
    gpu=${GPU_ARR[$(( k % ${#GPU_ARR[@]} ))]}
    { echo "CHROM,POS,REF,ALT"; cat "$f"; } > "$f.csv"
    ( CUDA_VISIBLE_DEVICES="$gpu" "$P/pangolin" -c CHROM,POS,REF,ALT \
        "$f.csv" "$FASTA" "$DB" "$f.out" > "$f.log" 2>&1 ) &
    pids+=($!); k=$((k+1))
  done
  for p in "${pids[@]}"; do wait "$p" || echo "  a shard exited non-zero"; done

  head -1 "$W"/part.000.out.csv > "$W/$task.pangolin.csv" 2>/dev/null || true
  for o in "$W"/part.*.out.csv; do tail -n +2 "$o"; done >> "$W/$task.pangolin.csv"
  "$P/python" parse_splice_vcf.py pangolin "$W/$task.pangolin.csv" \
      "scores/${task}__Pangolin.tsv"
done
echo "PANGOLIN_DONE $(date)"
