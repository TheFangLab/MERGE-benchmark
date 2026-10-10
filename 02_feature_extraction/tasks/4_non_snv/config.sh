#!/usr/bin/env bash
# Non-SNV variants (insertions, deletions, MNVs, delins)
TASK_ID="4_non_snv"
TASK_NAME="Non-SNV variants"
VARIANT_TABLE="03_datasets/non_snv_features.csv"
N_VARIANTS=36862

MODELS=(evo2_7b evo2_7b_base evo2_1b_base carbon generator generator_v2 nt_v2 \
        nt hyenadna enformer alphagenome_global)

# The big-library SNV scorers only handle single-base substitutions, so nt_old_score and
# hyenadna_score are computed on reconstructed haplotypes instead. Same checkpoints,
# different scoring convention — documented in FEATURE_METHODS.md rather than harmonised.
LM_PATH="haplotype"
NT_HAPLOTYPE_WINDOW_BP=600
NT_HAPLOTYPE_MAX_LENGTH=128
HYENADNA_HAPLOTYPE_WINDOW_BP=600
HYENADNA_HAPLOTYPE_MAX_LENGTH=1024

ALPHAGENOME_COLUMNS="global"

# Missense- and alignment-based scores are undefined for indels
PRECOMPUTED_LOOKUPS=()
