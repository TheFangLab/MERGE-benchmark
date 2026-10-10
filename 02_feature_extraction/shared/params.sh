#!/usr/bin/env bash
# Model-level parameters. These are identical for every task — that is the point of the
# manuscript's claim that the four datasets are scored with one algorithm and one set of
# hyper-parameters. Anything that legitimately differs between tasks lives in
# ../tasks/<task>/config.sh instead.
#
# Sourced by ../run_task.sh.

# --- checkpoints ---------------------------------------------------------------------
EVO2_7B_MODEL="arcinstitute/evo2_7b"
EVO2_7B_BASE_MODEL="arcinstitute/evo2_7b_base"
EVO2_1B_BASE_MODEL="arcinstitute/evo2_1b_base"
CARBON_MODEL="HuggingFaceBio/Carbon-3B"
GENERATOR_MODEL="GenerTeam/GENERator-eukaryote-1.2b-base"
GENERATOR_V2_MODEL="GenerTeam/GENERator-v2-eukaryote-1.2b-base"
NT_MODEL="InstaDeepAI/nucleotide-transformer-500m-human-ref"
NT_V2_MODEL="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
HYENADNA_MODEL="LongSafari/hyenadna-medium-160k-seqlen-hf"

# --- windows and tokenisation --------------------------------------------------------
EVO2_WINDOW_BP=8192            # centred; mean_PLL(ref) - mean_PLL(alt)
EVO2_AVERAGE_RC=false          # average_reverse_complement=False
CARBON_WINDOW_BP=24576         # logP_sum(ref) - logP_sum(alt), SUM-delta
VARIANT_LM_WINDOW_BP=600       # generator, generator_v2, nt_v2
VARIANT_LM_MAX_LENGTH=128
VARIANT_LM_BATCH_SIZE=16      # SNV windows; haplotype scoring uses 8. float32 results do not
                              # depend on it beyond 3e-5
NT_SNV_WINDOW_BP=8192          # NT_score on the SNV tasks
NT_SNV_MAX_LENGTH=1000         # = tokenizer.model_max_length
HYENADNA_SNV_CONTEXT_BP=1000   # upstream ref context, next-token log-prob delta
ALPHAGENOME_SEQUENCE_LENGTH=1048576   # 2^20

# --- reference -----------------------------------------------------------------------
FASTA="${FASTA:-$HOME/genomes/hg38.fa}"
