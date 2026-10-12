# 03_datasets — the four analysis-ready feature tables

These are the authoritative modelling inputs. Everything in `04_modeling/` reads exactly
these files, and every number in the manuscript derives from them.

| File | Variants | Positive / negative | Feature columns |
|---|---:|---:|---:|
| `coding_features.csv` | 4,289 | 2,912 / 1,377 | 25 |
| `splicing_features.csv` | 35,015 | 22,247 / 12,768 | 18 |
| `noncoding_features.csv` | 86,829 | 1,340 / 85,489 | 23 |
| `non_snv_features.csv` | 36,862 | 30,022 / 6,840 | 22 |

Every table starts with the same five identifier columns — `chr, pos, ref, alt, label` —
followed by the feature columns of that task. `chr` has no `chr` prefix; coordinates are
hg38, one-based, VCF convention. `label` is 1 for pathogenic / likely pathogenic (and, in
the splicing task, experimentally splice-altering) and 0 for benign / likely benign,
normal-splicing or common population variants.

## Feature columns by task

**Shared by all four tasks (16 columns)**
`heyenadna_score`/`hyenadna_score`, `NT_score`/`nt_old_score`, `nt_v2_score`,
`carbon_score`, `evo2_7b_score`, `evo2_7b_base_score`, `evo2_1b_base_score`,
`generator_score`, `generator_v2_score`, and the seven Enformer summaries
(`enformer_absmean`, `enformer_mean`, `enformer_std`, `enformer_max_abs`,
`enformer_top5_absmean`, `enformer_pos_sum`, `enformer_neg_sum`).

**Task-specific**

| Task | Adds |
|---|---|
| coding | `AlphaMissense_score`, `ESM1b_score`, `GPN_MSA_score`, six AlphaGenome global summaries |
| splicing | `GPN_MSA_score`, `alphagenome_splicing` |
| noncoding | `GPN_MSA_score`, six AlphaGenome global summaries |
| non_snv | six AlphaGenome global summaries (no missense or alignment-based scores — they are undefined for indels) |

The six AlphaGenome global summaries are `alphagenome_{raw,quantile}_score_{max,min,mean}`.

## Naming

Two columns carry different names in the SNV and non-SNV tables because the scoring
convention differs, although the model is the same:

| SNV tables | non-SNV table | Model |
|---|---|---|
| `NT_score` | `nt_old_score` | `InstaDeepAI/nucleotide-transformer-500m-human-ref` |
| `heyenadna_score` | `hyenadna_score` | `LongSafari/hyenadna-medium-160k-seqlen-hf` |

See `../02_feature_extraction/README.md` for why.

## Missing values

`AlphaMissense_score`, `ESM1b_score` and `GPN_MSA_score` have incomplete coverage where no
official precomputed score matched the exact allele, and `alphagenome_splicing` is missing
for 571 splicing variants. All missing values are imputed with the **training-fold**
median inside the cross-validation, never globally.

`GPN_MSA_score` is matched to the official genome-wide GPN-MSA table by chromosome,
position, REF and ALT (`02_feature_extraction/shared/match_gpn_msa.py`). In the splicing
table, 6,796 values in one contiguous block of rows (rows 14,120–20,924 of the file) had
been attached to the wrong variants when the table was first assembled; they were
re-matched on 8 October 2026, and every splicing result in this repository uses the
corrected column. Every value in rows 13,000–22,000 was checked against the official
table; samples of the rest of the splicing table and of the coding and noncoding tables
(about 1,900 variants, plus every splicing and coding variant shared with the
precomputed candidate sets) matched exactly, and no other feature column of the block
was affected.

`generator_score` and `generator_v2_score` are float32 scores; a re-run with the scripts in
`02_feature_extraction/` reproduces them exactly at the same batch size and to 5e-5 at any
other. The non-SNV `hyenadna_score` is scored one variant at a time (float32), because
HyenaDNA has no attention mask and batch padding would change it. Evo 2, Carbon and Enformer
reproduce closely but not bit for bit; AlphaGenome follows the hosted model, which has changed
since the coding and noncoding columns were computed. Measured agreements:
`02_feature_extraction/README.md`.

Three groups of columns were recomputed on 12 October 2026: the seven `enformer_*` columns of
the splicing and noncoding tables, which had covered only 4,047 of Enformer's 5,313 tracks
(every table now uses all 5,313), and the six `alphagenome_*` columns of the non-SNV table
and of the BRCA1 SGE features (`../08_external_validation/features/`), whose earlier values
could not be reproduced with the current AlphaGenome API. All other columns are unchanged.

The splicing table is the strict deduplication of the three-source assembly: 35,774 rows,
of which every row of the 379 duplicated (chr, pos, ref, alt) keys (759 rows) is removed
(`01_dataset_construction/coding_noncoding_splicing/07_strict_deduplicate_splicing.py`,
audit in `splicing_strict_dedup_audit.csv` next to it).

## Licence

These tables are released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); cite the manuscript.

The labels derive from ClinVar, gnomAD v4.1 and SpliceVarDB, and some feature columns come
from dbNSFP and the AlphaGenome API, all of which keep their own terms — dbNSFP requires a
separate licence for commercial use and the AlphaGenome API is offered for non-commercial
use. See the licence section of the top-level README.
