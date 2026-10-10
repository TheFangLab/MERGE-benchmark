# Non-SNV variants (indel / MNV / delins)

**Dataset:** `non_snv_features.csv` — 36,862 variants. Identifier columns: `chr,pos,ref,alt,label`.

All feature columns use unified names and unified computation across the four datasets (see top-level README.md).

## Features and extraction methods

Non-SNV variants are scored on full REF/ALT **haplotype** sequences rebuilt from hg38 (VCF anchor prefix removed, real edit located, same up/down-stream window kept). See `../../shared/non_snv_sequences.py`.

| feature | model | computation | code |
|---|---|---|---|
| generator_v2_score | GENERator-v2-1.2b-base (causal) | mean_NLL(alt) - mean_NLL(ref), 600 bp haplotype, max_length=128, float32 (batch size 8). | `../../shared/non_snv_lm_scores.py` |
| generator_score | GENERator-1.2b-base (causal) | mean_NLL(alt) - mean_NLL(ref), 600 bp, max_length=128, float32 (batch size 8). | `../../shared/non_snv_lm_scores.py` |
| nt_old_score | NT old (`nucleotide-transformer-500m-human-ref`, masked) | ‖emb_alt - emb_ref‖₂ on REF/ALT **haplotypes**, 600 bp, max_length=128. ⚠️ **Deliberately not the SNV big-library NT method** (which is 8192 bp / max_length=1000 single-base substitution): that path is SNV-only and cannot score indels/MNVs, so the haplotype masked-L2 variant is kept here. Same "old NT" model as the SNV datasets' `NT_score`. | `../../shared/non_snv_lm_scores.py` |
| nt_v2_score | NT-v2 500M (masked) | ||emb_alt - emb_ref||_2 on haplotypes, 600 bp, max_length=128. | `../../shared/non_snv_lm_scores.py` |
| hyenadna_score | HyenaDNA (char-level, causal) | mean_NLL(alt) - mean_NLL(ref) on REF/ALT **haplotypes**, genomic window=600, max_length=1024, float32, batch size 1 (HyenaDNA has no attention mask, so padding in a batch would change the scores). ⚠️ **Deliberately not the SNV big-library HyenaDNA method** (upstream-context next-token `log p(alt) − log p(ref)`): the next-token formula does not extend to indels/MNVs, so the haplotype mean-NLL variant is kept here. Same HyenaDNA model as the SNV datasets' `heyenadna_score`. | `../../shared/non_snv_lm_scores.py` |
| carbon_score | Carbon-3B | logP_sum(ref) - logP_sum(alt) over a 24,576 bp window; DNA wrapped as <dna>...</dna>. (SUM-delta convention, identical across all four datasets.) (== `carbon_3b_score_sum` from the shard output, chosen to match the SNV datasets' SUM-delta carbon_score). | `../../shared/non_snv_causal_features.py` |
| alphagenome_raw/quantile_score_{max,min,mean} | AlphaGenome | dna_client + RECOMMENDED_VARIANT_SCORERS, 2^20 bp; per-scorer stats + cross-scorer aggregation. | `../../shared/alphagenome_non_snv.py` + `../../shared/summarize_alphagenome.py` |
| evo2_7b_base_score / evo2_7b_score / evo2_1b_base_score | Evo2 7B-base / 7B / 1B-base | mean_PLL(ref) - mean_PLL(alt); 8192 bp window centered on the variant; average_reverse_complement=False. Larger => alt less likely under the model (more deleterious). (== `evo2_*_score_mean` from the shard output; identical formula to the SNV datasets). | `../../shared/non_snv_causal_features.py` |
| enformer_* (7) | Enformer | 7 summary stats over Enformer variant tracks (alt-ref, per-track mean over bins, then normalized): absmean, mean, std, max_abs, top5_absmean, pos_sum, neg_sum. Same script/computation for all four datasets. Computed fresh for the 36,862 non-SNV variants with the same `enformer_summary_shard.py` used for the SNV datasets (kipoiseq VariantSeqExtractor handles indels/MNVs). | `../../shared/enformer_summary_shard.py` |

MT-chromosome variants are excluded: AlphaGenome does not score chrM.
