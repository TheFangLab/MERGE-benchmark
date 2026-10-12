#!/usr/bin/env bash
# Print the per-model scoring commands for one task.
#
#   bash run_task.sh 2_splicing              # print the command manifest
#   bash run_task.sh 2_splicing --variants my_variants.csv --outdir ./out
#
# This prints; it does not execute. Every one of these commands needs a GPU host, a local
# hg38 FASTA, local Hugging Face checkpoints and (for AlphaGenome) an API key, and the
# heavier ones were run sharded across 6-8 GPUs for days. The manifest exists so a reader
# can see exactly which script, which flags and which window produced each column of
# ../03_datasets/<task>_features.csv without diffing four copies of the same directory.
#
# The parameters come from shared/params.sh (identical for every task) and
# tasks/<task>/config.sh (what legitimately differs).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK="${1:-}"
shift || true

if [[ -z "${TASK}" || ! -f "${HERE}/tasks/${TASK}/config.sh" ]]; then
  echo "usage: bash run_task.sh <task> [--variants FILE] [--outdir DIR]" >&2
  echo "tasks:" >&2
  for d in "${HERE}"/tasks/*/; do echo "  $(basename "${d}")" >&2; done
  exit 2
fi

# shellcheck source=/dev/null
source "${HERE}/shared/params.sh"
# shellcheck source=/dev/null
source "${HERE}/tasks/${TASK}/config.sh"

VARIANTS=""
OUTDIR="./out/${TASK}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --variants) VARIANTS="$2"; shift 2 ;;
    --outdir)   OUTDIR="$2";   shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
VARIANTS="${VARIANTS:-${HERE}/../${VARIANT_TABLE}}"

S="\${SHARED}"
cat <<HEADER
# ${TASK_NAME} — ${N_VARIANTS} variants
# variant table : ${VARIANTS}
# output dir    : ${OUTDIR}
# shared code   : ${HERE}/shared
# LM path       : ${LM_PATH}

SHARED=${HERE}/shared
IN=${VARIANTS}
OUT=${OUTDIR}
FASTA=${FASTA}
mkdir -p "\$OUT"

HEADER

emit() { printf '%s\n\n' "$1"; }

# SNV tasks score a single-base substitution in a window centred on the variant; the
# non-SNV task (LM_PATH=haplotype) scores full REF/ALT haplotypes rebuilt from hg38 with
# the non_snv_* scripts, because the SNV scripts skip anything that is not one base.
HAP=false
[[ "${LM_PATH}" == "haplotype" ]] && HAP=true

for model in "${MODELS[@]}"; do
  case "${model}" in
    evo2_7b|evo2_7b_base|evo2_1b_base)
      if ${HAP}; then
        emit "# ${model}_score <- column ${model}_score_mean: mean_logP(ref) - mean_logP(alt), ${EVO2_WINDOW_BP} bp haplotypes
python3 ${S}/non_snv_causal_features.py --backend evo2 --model-name ${model} \\
  --input-csv \"\$IN\" --output-csv \"\$OUT/${model}.csv\" --fasta \"\$FASTA\" \\
  --window-bp ${EVO2_WINDOW_BP} --resume"
      else
        emit "# ${model}_score <- column evo2_score: mean_PLL(ref) - mean_PLL(alt), ${EVO2_WINDOW_BP} bp, no reverse-complement averaging
python3 ${S}/precompute_evo2_vep_fixed.py --model-name ${model} \\
  --input-csv \"\$IN\" --output-csv \"\$OUT/${model}.csv\" --fasta \"\$FASTA\" \\
  --window-bp ${EVO2_WINDOW_BP}"
      fi ;;
    carbon)
      if ${HAP}; then
        emit "# carbon_score <- column carbon_3b_score_sum: logP_sum(ref) - logP_sum(alt), ${CARBON_WINDOW_BP} bp haplotypes
python3 ${S}/non_snv_causal_features.py --backend carbon \\
  --input-csv \"\$IN\" --output-csv \"\$OUT/carbon.csv\" --fasta \"\$FASTA\" \\
  --window-bp ${CARBON_WINDOW_BP} --resume"
      else
        emit "# carbon_score = logP_sum(ref) - logP_sum(alt), ${CARBON_WINDOW_BP} bp; one shard per GPU, then merged
python3 ${S}/precompute_carbon_scores.py \\
  --input-csv \"\$IN\" --output-csv \"\$OUT/carbon_shard\${SHARD}.csv\" \\
  --fasta \"\$FASTA\" --model-id ${CARBON_MODEL} \\
  --window-bp ${CARBON_WINDOW_BP} --shard-index \${SHARD} --num-shards \${NUM_SHARDS}
python3 ${S}/merge_carbon_shards.py --shard-dir \"\$OUT\" --pattern 'carbon_shard*.csv' \\
  --output-csv \"\$OUT/carbon.csv\""
      fi ;;
    generator|generator_v2|nt_v2)
      case "${model}" in
        generator)    ckpt="${GENERATOR_MODEL}";    kind=causal; what="mean_NLL(alt) - mean_NLL(ref)" ;;
        generator_v2) ckpt="${GENERATOR_V2_MODEL}"; kind=causal; what="mean_NLL(alt) - mean_NLL(ref)" ;;
        *)            ckpt="${NT_V2_MODEL}";        kind=masked; what="||emb(alt) - emb(ref)||_2" ;;
      esac
      ${HAP} && script=non_snv_lm_scores.py || script=variant_lm_scores.py
      ${HAP} && batch=8 || batch=${VARIANT_LM_BATCH_SIZE}
      emit "# ${model}_score = ${what}, ${VARIANT_LM_WINDOW_BP} bp, max_length ${VARIANT_LM_MAX_LENGTH}, float32
python3 ${S}/${script} --model-kind ${kind} --model ${ckpt} --trust-remote-code \\
  --input-csv \"\$IN\" --output-csv \"\$OUT/${model}.csv\" --fasta \"\$FASTA\" \\
  --window-bp ${VARIANT_LM_WINDOW_BP} --max-length ${VARIANT_LM_MAX_LENGTH} \\
  --batch-size ${batch} --score-col ${model}_score" ;;
    nt)
      if ${HAP}; then
        emit "# nt_old_score = ||emb(alt) - emb(ref)||_2 on haplotypes, ${NT_HAPLOTYPE_WINDOW_BP} bp, max_length ${NT_HAPLOTYPE_MAX_LENGTH}
python3 ${S}/non_snv_lm_scores.py --model-kind masked --model ${NT_MODEL} \\
  --input-csv \"\$IN\" --output-csv \"\$OUT/nt_old.csv\" --fasta \"\$FASTA\" \\
  --window-bp ${NT_HAPLOTYPE_WINDOW_BP} --max-length ${NT_HAPLOTYPE_MAX_LENGTH} \\
  --score-col nt_old_score"
      else
        emit "# NT_score = ||emb(alt) - emb(ref)||_2, ${NT_SNV_WINDOW_BP} bp, max_length ${NT_SNV_MAX_LENGTH}
python3 ${S}/score_bigfib.py --model nt --input \"\$IN\" --output \"\$OUT/nt.csv\" \\
  --fasta \"\$FASTA\""
      fi ;;
    hyenadna)
      if ${HAP}; then
        emit "# hyenadna_score = mean_NLL(alt) - mean_NLL(ref) on haplotypes, ${HYENADNA_HAPLOTYPE_WINDOW_BP} bp, max_length ${HYENADNA_HAPLOTYPE_MAX_LENGTH}
python3 ${S}/non_snv_lm_scores.py --model-kind causal --model ${HYENADNA_MODEL} --trust-remote-code \\
  --input-csv \"\$IN\" --output-csv \"\$OUT/hyenadna.csv\" --fasta \"\$FASTA\" \\
  --window-bp ${HYENADNA_HAPLOTYPE_WINDOW_BP} --max-length ${HYENADNA_HAPLOTYPE_MAX_LENGTH} \\
  --batch-size 1 --score-col hyenadna_score"
      else
        emit "# heyenadna_score = log p(alt) - log p(ref), ${HYENADNA_SNV_CONTEXT_BP} bp upstream context
python3 ${S}/score_bigfib.py --model hyena --input \"\$IN\" --output \"\$OUT/hyenadna.csv\" \\
  --fasta \"\$FASTA\""
      fi ;;
    enformer)
      emit "# enformer_* (7 summaries over all 5,313 human tracks); one shard of rows [START, END) per GPU, then merged
python3 ${S}/enformer_summary_shard.py --gpu \${GPU} --start-index \${START} --end-index \${END} \\
  --input-csv \"\$IN\" --output-csv \"\$OUT/enformer_gpu\${GPU}.csv\"
python3 ${S}/merge_enformer_summary_shards.py \\
  \$(for f in \"\$OUT\"/enformer_gpu*.csv; do printf -- '--shard-output %s ' \"\$f\"; done) \\
  --final-output \"\$OUT/enformer.csv\"" ;;
    alphagenome_global)
      emit "# alphagenome_{raw,quantile}_score_{max,min,mean}: per-scorer statistics over a ${ALPHAGENOME_SEQUENCE_LENGTH} bp window, then aggregated (API key required)
python3 ${S}/alphagenome_non_snv.py --input-csv \"\$IN\" --output-csv \"\$OUT/alphagenome_scorers.csv\" \\
  --sequence-length ${ALPHAGENOME_SEQUENCE_LENGTH} --resume
python3 ${S}/summarize_alphagenome.py --input-csv \"\$OUT/alphagenome_scorers.csv\" \\
  --output-csv \"\$OUT/alphagenome.csv\"" ;;
    alphagenome_splicing)
      emit "# alphagenome_splicing = max(splice_sites) + max(splice_site_usage) + max(splice_junctions)/5, 1 Mb (API key required)
python3 ${S}/alphagenome_splicing.py --input-csv \"\$IN\" --output-csv \"\$OUT/alphagenome_splicing.csv\" --resume" ;;
  esac
done


if [[ ${#PRECOMPUTED_LOOKUPS[@]} -gt 0 ]]; then
  printf '# joined from official precomputed tables by (chr, pos, ref, alt) — no inference script:\n'
  printf '#   %s\n' "${PRECOMPUTED_LOOKUPS[@]}"
  printf '# sources: ../00_raw_data/README.md\n'
fi
