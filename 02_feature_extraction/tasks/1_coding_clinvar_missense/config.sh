#!/usr/bin/env bash
# Coding missense SNVs
TASK_ID="1_coding_clinvar_missense"
TASK_NAME="Coding missense SNVs"
VARIANT_TABLE="03_datasets/coding_features.csv"
N_VARIANTS=4289

# Models that contribute a column to this task
MODELS=(evo2_7b evo2_7b_base evo2_1b_base carbon generator generator_v2 nt_v2 \
        nt hyenadna enformer alphagenome_global)

# NT_score / heyenadna_score scoring path: the single-base-substitution
# "big-library" scorer (shared/score_bigfib.py). Identical across the
# three SNV tasks, which is why those two columns match between them.
LM_PATH="snv_biglibrary"

# AlphaGenome columns: the six cross-scorer summaries
ALPHAGENOME_COLUMNS="global"

# Columns joined from official precomputed tables — no inference script exists
PRECOMPUTED_LOOKUPS=(AlphaMissense_score ESM1b_score GPN_MSA_score)
