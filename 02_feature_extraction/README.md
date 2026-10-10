# 02_feature_extraction — per-model scoring code

One copy of each script, plus a small configuration file per task.

```
shared/               every scoring script, exactly once, + params.sh
tasks/
  1_coding_clinvar_missense/  FEATURE_METHODS.md  config.sh
  2_splicing/                 FEATURE_METHODS.md  config.sh
  3_noncoding/                FEATURE_METHODS.md  config.sh
  4_non_snv/                  FEATURE_METHODS.md  config.sh
run_task.sh           prints the per-model command manifest for one task
```

The three SNV tasks are scored with one algorithm and one set of hyper-parameters, so the
scripts live once in `shared/` rather than being copied per task. `tasks/*/config.sh` is
the complete list of what is task-specific.

## What differs between tasks

Everything, and only what is in `tasks/<task>/config.sh`:

| Variable | What it captures |
|---|---|
| `VARIANT_TABLE`, `N_VARIANTS` | which variant list is scored |
| `MODELS` | which models contribute a column — non-SNV has no missense or alignment-based score, splicing takes the AlphaGenome splice combination rather than the six global summaries |
| `LM_PATH` | `snv_biglibrary` for the three SNV tasks, `haplotype` for non-SNV (see below) |
| `ALPHAGENOME_COLUMNS` | `global` (six summaries) or `splicing` (one combined column) |
| `PRECOMPUTED_LOOKUPS` | official score tables joined by `(chr, pos, ref, alt)`; AlphaMissense and ESM-1b exist only for coding, GPN-MSA for the three SNV tasks |

Model-level parameters — windows, `max_length`, checkpoints — are identical for every
task and therefore live once, in `shared/params.sh`.

## Seeing what produced a column

```bash
bash run_task.sh 2_splicing
```

prints the scoring command for each model of that task, with the flags and windows filled
in from the two config files. It prints; it does not execute — each command needs a GPU
host, a local hg38 FASTA, local Hugging Face checkpoints and, for AlphaGenome, an API key,
and the heavy ones were sharded across six to eight GPUs for days.

`tasks/<task>/FEATURE_METHODS.md` remains the per-column reference: model, window, score
formula, and the script in `shared/` that produces it.

## Which script produces which feature

| Feature column(s) | Script | Notes |
|---|---|---|
| `evo2_7b_score`, `evo2_7b_base_score`, `evo2_1b_base_score` | SNV: `precompute_evo2_vep_fixed.py`; non-SNV: `non_snv_causal_features.py --backend evo2` | 8,192 bp centred window, `mean_PLL(ref) − mean_PLL(alt)`, no reverse-complement averaging |
| `carbon_score` | `precompute_carbon_scores.py` + `merge_carbon_shards.py` | 24,576 bp window, `logP_sum(ref) − logP_sum(alt)` (SUM-delta) |
| `generator_score`, `generator_v2_score`, `nt_v2_score` | SNV: `variant_lm_scores.py`; non-SNV: `non_snv_lm_scores.py` | 600 bp / 128 tokens; causal NLL difference, or L2 distance between pooled embeddings; float32 unless `--bf16` |
| `NT_score` (SNV) | `score_bigfib.py` | 8,192 bp window, `max_length = 1000`, L2 embedding distance |
| `heyenadna_score` (SNV) | `score_bigfib.py` | 1,000 bp upstream context, next-token `log p(alt) − log p(ref)` |
| `enformer_*` (7 columns) | `enformer_summary_shard.py` + `merge_enformer_summary_shards.py` | 4,047 tracks summarized as absmean / mean / std / max_abs / top5_absmean / pos_sum / neg_sum |
| `alphagenome_{raw,quantile}_score_{max,min,mean}` | `alphagenome_non_snv.py` + `summarize_alphagenome.py` | `dna_client` + `RECOMMENDED_VARIANT_SCORERS`, 2²⁰ bp window; per-scorer statistics, then max / min / count-weighted mean over 11 scorers |
| `alphagenome_splicing` | `alphagenome_splicing.py` | `max(splice_sites) + max(splice_site_usage) + max(splice_junctions) / 5`, 1 Mb interval |
| Non-SNV `carbon_score` | `non_snv_causal_features.py --backend carbon` | 24,576 bp haplotypes, SUM-delta |
| Non-SNV `nt_old_score`, `hyenadna_score` | `non_snv_lm_scores.py` | haplotype scoring; see the deliberate exception below |
| `GPN_MSA_score` | `match_gpn_msa.py` | official precomputed table, joined by `(chr, pos, ref, alt)` |
| `ESM1b_score`, `AlphaMissense_score` | — | dbNSFP 5.3a, joined by `(chr, pos, ref, alt)` (mean over transcripts); no inference script exists |

## Deliberate exception for non-SNV NT and HyenaDNA

The big-library NT and HyenaDNA scorers only handle single-base substitutions. For
non-SNV variants, `nt_old_score` (masked L2 distance, 600 bp, `max_length=128`) and
`hyenadna_score` (causal `mean_NLL(alt) − mean_NLL(ref)`, 600 bp, `max_length=1024`)
therefore keep the haplotype implementation in `non_snv_lm_scores.py`. Same models,
different scoring convention; this is documented rather than silently harmonized, and it
is why `tasks/4_non_snv/config.sh` sets `LM_PATH=haplotype`.

## Reproducing the stored values

The precomputed scores released with MERGE were produced with these scripts, so the
variants they share with the benchmark tables show how closely a re-run reproduces a
stored column (Pearson r; splicing n = 4,804, coding n = 330):

| Column | Splicing | Coding | |
|---|---:|---:|---|
| `GPN_MSA_score`, `nt_v2_score` | 1.0000 | 1.0000 | identical, or equal to 1e-5 |
| `heyenadna_score` | — | 1.0000 | equal to 1e-5 |
| six global `alphagenome_*` | — | 0.9995–1.0000 | equal to 1e-9 for at least 98.5% of variants; quantile mean differs by a median 8e-5 |
| `ESM1b_score`, `AlphaMissense_score` | — | 0.995–0.9997 | dbNSFP lookup; 99.4–99.7% identical |
| `alphagenome_splicing` | 0.9999 | — | hosted model; median absolute difference 0.002 |
| `evo2_*` | 0.998–0.9996 | 0.97–0.99 | GPU and bfloat16 nondeterminism |
| `carbon_score` | 0.9995 | 0.964 | as above |
| `enformer_*` | 0.997–0.999 | 0.9999–1.0000 | TensorFlow GPU nondeterminism |
| `generator_score`, `generator_v2_score` | 1.0000 | 1.0000 | identical; float32, see below |

The coding overlap is common variants with weak effects, so the same small numerical
noise lowers the correlation more there.

**GENERATOR and GENERATOR-v2 are computed in float32.** The score is a difference
between two mean NLLs that share all but one position, so reduced precision matters
here: in bfloat16 it picks up rounding noise of SD ≈ 0.04 (measured against float32 on
300 coding and 300 noncoding SNVs) that also changes with batch size and hardware (two
bfloat16 runs on the same GPU, batch size 1 versus 16, agree only at r = 0.95–0.97). In float32 the score is deterministic:
batch sizes 1, 8 and 16 agree to 5e-5. The columns in `03_datasets/` were computed with
`variant_lm_scores.py` (SNVs, batch size 16) and `non_snv_lm_scores.py` (non-SNV
haplotypes, batch size 8) on an RTX A6000, torch 2.7.1, transformers 4.57.6, Hugging
Face revisions `5e872c94` (GENERator-eukaryote-1.2b-base) and `c41b0018`
(GENERator-v2-eukaryote-1.2b-base).

**HyenaDNA on non-SNV haplotypes is scored one variant at a time.** HyenaDNA has no
attention mask, so in a padded batch the padding of shorter haplotypes changes the scores
of the others (at batch size 8, half of the scores depend on their batch neighbours).
The non-SNV `hyenadna_score` column was computed with `non_snv_lm_scores.py --batch-size 1`
in float32, which the script now enforces for HyenaDNA. The other language models use
the attention mask, and their scores do not depend on the batch size (NT-v2 at batch
sizes 1 and 8 agrees to 1e-5).

## Feature coverage

`AlphaMissense_score` and `ESM1b_score` exist only for the coding task; `GPN_MSA_score`
only for the three SNV tasks. Missing values are imputed with the training-fold median
inside the cross-validation.

## Running any of this

These scripts were written for specific GPU hosts and expect a local hg38 FASTA, local
Hugging Face checkpoints, and (for AlphaGenome) an API key. They are included as the
record of how each feature column was computed, not as a turnkey pipeline. The
analysis-ready output is committed in [`../03_datasets/`](../03_datasets/).
