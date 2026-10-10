#!/usr/bin/env bash
# Splicing SNVs
TASK_ID="2_splicing"
TASK_NAME="Splicing SNVs"
VARIANT_TABLE="03_datasets/splicing_features.csv"
N_VARIANTS=35774   # scored before strict deduplication; 35,015 in 03_datasets

# Models that contribute a column to this task
MODELS=(evo2_7b evo2_7b_base evo2_1b_base carbon generator generator_v2 nt_v2 \
        nt hyenadna enformer alphagenome_splicing)

# NT_score / heyenadna_score scoring path: the single-base-substitution
# "big-library" scorer (shared/score_bigfib.py). Identical across the
# three SNV tasks, which is why those two columns match between them.
LM_PATH="snv_biglibrary"

# AlphaGenome columns: the single splice combination (see FEATURE_METHODS.md)
ALPHAGENOME_COLUMNS="splicing"

# Columns joined from official precomputed tables — no inference script exists
PRECOMPUTED_LOOKUPS=(GPN_MSA_score)
