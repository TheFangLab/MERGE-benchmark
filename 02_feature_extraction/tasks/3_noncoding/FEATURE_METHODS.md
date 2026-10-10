# Non-coding SNVs

**Dataset:** `noncoding_features.csv` — 86,829 variants. Identifier columns: `chr,pos,ref,alt,label`.

All feature columns use unified names and unified computation across the four datasets (see top-level README.md).

## Features and extraction methods

| feature | model | computation | code |
|---|---|---|---|
| GPN_MSA_score | GPN-MSA | official genome-wide hg38 score (songlab/gpn-msa-hg38-scores), matched on chr, pos, ref and alt. | `../../shared/match_gpn_msa.py` |
| NT_score | NT — old (`nucleotide-transformer-500m-human-ref`, masked) | ‖emb(alt) − emb(ref)‖₂ over a 8192 bp window centered on the variant; mean-pooled last hidden; max_length=model_max_length (1000). Recomputed fresh with the big-library code; identical across all three SNV datasets. | `../../shared/score_bigfib.py --model nt` |
| nt_v2_score | NT-v2 (masked) | ||emb_alt - emb_ref||_2, 600 bp, max_length=128. | `../../shared/variant_lm_scores.py --model-kind masked` |
| heyenadna_score | HyenaDNA (local, char-level causal) | next-token `log p(alt) − log p(ref)` over a 1000 bp **upstream** ref context. Recomputed fresh with the big-library code; identical across all three SNV datasets. | `../../shared/score_bigfib.py --model hyena` |
| alphagenome_raw/quantile_score_{max,min,mean} | AlphaGenome | dna_client + RECOMMENDED_VARIANT_SCORERS, 2^20 bp; per-scorer stats + cross-scorer aggregation. | `../../shared/alphagenome_non_snv.py` + `../../shared/summarize_alphagenome.py` |
| carbon_score | Carbon-3B | logP_sum(ref) - logP_sum(alt) over a 24,576 bp window; DNA wrapped as <dna>...</dna>. (SUM-delta convention, identical across all four datasets.) | `../../shared/precompute_carbon_scores.py` |
| enformer_* (7) | Enformer | 7 summary stats over Enformer variant tracks (alt-ref, per-track mean over bins, then normalized): absmean, mean, std, max_abs, top5_absmean, pos_sum, neg_sum. Same script/computation for all four datasets. | `../../shared/enformer_summary_shard.py` |
| evo2_7b_score / evo2_7b_base_score / evo2_1b_base_score | Evo2 7B / 7B-base / 1B-base | mean_PLL(ref) - mean_PLL(alt); 8192 bp window centered on the variant; average_reverse_complement=False. Larger => alt less likely under the model (more deleterious). | `../../shared/precompute_evo2_vep_fixed.py` |
| generator_score / generator_v2_score | GENERator / GENERator-v2 (causal) | NLL(alt) - NLL(ref), 600 bp, max_length=128, float32 (batch size 16). | `../../shared/variant_lm_scores.py --model-kind causal` |
