# MERGE precomputed scores (hg38)

Precomputed MERGE scores and the underlying model-derived features for two candidate
sets of single-nucleotide variants (SNVs), from *Benchmarking genomic foundation and
sequence-to-function models for variant pathogenicity prediction* (Chen and Fang).

| File | Variants | Content |
|---|---:|---|
| `coding_precompute_MERGE_scored.csv.gz` | 107,922 | coding SNVs: MERGE-Coding score and features |
| `splicing_precompute_MERGE_scored.csv.gz` | 1,920,006 | splice-region SNVs: MERGE-Splice score, the MERGE + AlphaGenome average and features |
| `MD5SUMS.txt` | | checksums |

## Download

The two `.csv.gz` files are larger than GitHub allows in a repository (100 MB), so they
are attached to the release
[`v1.0`](https://github.com/XiaoYangGao0401/MERGE-precomputed-scores/releases/tag/v1.0)
of [XiaoYangGao0401/MERGE-precomputed-scores](https://github.com/XiaoYangGao0401/MERGE-precomputed-scores).
That repository and `09_precomputed_scores/` of
[MERGE-benchmark](https://github.com/TheFangLab/MERGE-benchmark) keep this README and the
checksums.

```bash
B=https://github.com/XiaoYangGao0401/MERGE-precomputed-scores/releases/download/v1.0
curl -LO $B/coding_precompute_MERGE_scored.csv.gz      # 21 MB
curl -LO $B/splicing_precompute_MERGE_scored.csv.gz    # 274 MB
md5sum -c MD5SUMS.txt
```

## Candidate sets

* **Splice region**: all possible SNVs in GENCODE v44 exon–intron boundaries extended by
  2 bp (BEDTools), enumerated with SAMtools/BCFtools and annotated with ANNOVAR:
  1,920,006 SNVs.
* **Coding**: all possible SNVs in GENCODE v44 coding sequences (104,108,391), minus
  synonymous variants and duplicate amino-acid changes (69,708,678 left), restricted to
  variants present in gnomAD with allele frequency > 0.001: 107,922 SNVs. This set is enriched for observed common
  variants; it is not a catalogue of rare or clinically relevant variation. Other
  variants can be scored with the web server or merge-cli.

Coordinates are hg38, one-based, VCF-style; `chr` carries the `chr` prefix. For hg19, use
UCSC liftOver with the hg38-to-hg19 chain.

## Columns

| Column | Description |
|---|---|
| `chr`, `pos`, `ref`, `alt` | variant (hg38) |
| `evo2_7b_score`, `evo2_7b_base_score`, `evo2_1b_base_score` | Evo 2: mean log-likelihood(REF) − mean log-likelihood(ALT), 8,192-bp window |
| `carbon_score` | Carbon-3B: summed log-likelihood(REF) − summed log-likelihood(ALT), 24,576-bp window |
| `generator_score`, `generator_v2_score` | GENERATOR / GENERATOR-v2: mean NLL(ALT) − mean NLL(REF), 600 bp |
| `nt_v2_score` | Nucleotide Transformer v2: L2 distance between mean-pooled REF and ALT embeddings, 600 bp |
| `heyenadna_score` (coding) | HyenaDNA: log p(ALT) − log p(REF) given 1,000 bp of upstream context |
| `enformer_absmean` … `enformer_neg_sum` | seven summaries of the Enformer ALT − REF effects over all 5,313 human tracks |
| `GPN_MSA_score` | official GPN-MSA score |
| `AlphaMissense_score`, `ESM1b_score` (coding) | from dbNSFP 5.3a (mean over transcripts) |
| `alphagenome_*` (coding: six summaries; splicing: `alphagenome_splicing`) | AlphaGenome variant-effect summaries (API) |
| `MERGE_score` | MERGE probability from the task's released model |
| `MERGE_alphagenome_average` (splicing) | 0.5 × `MERGE_score` + 0.5 × the percentile of `alphagenome_splicing` among the benchmark splicing variants; empty when AlphaGenome has no score |

Feature definitions follow the manuscript Methods and the MERGE-benchmark repository
(`02_feature_extraction/`). An empty feature cell means the model returned no score
(for example a window containing N, or no dbNSFP or GPN-MSA entry); the MERGE models then
use the training median for that feature, as they do on the web server and in merge-cli.

## Models

`MERGE_score` and `MERGE_alphagenome_average` come from the released MERGE models
(MERGE-benchmark `05_results/trained_models/`, identical to those in merge-cli
3.18.0): MERGE-Coding (beam search, L2 logistic regression, 6 features) and
MERGE-Splice (LASSO, CatBoost, 12 features). Neither model uses an
AlphaGenome feature. Each file can be re-scored exactly from its feature columns with
those models.

```python
import pandas as pd, joblib
df = pd.read_csv("splicing_precompute_MERGE_scored.csv.gz")
m = joblib.load("05_results/trained_models/MERGE_noAlphaGenome_splicing.pkl")   # in MERGE-benchmark
score = m["pipeline"].predict_proba(df[m["features"]].to_numpy())[:, 1]   # equals df.MERGE_score
```

Score bands used by the web server and merge-cli for `MERGE_score`: ≥ 0.8 Pathogenic,
≥ 0.6 Likely Pathogenic, ≥ 0.4 Uncertain Significance, ≥ 0.2 Likely Benign, otherwise
Benign. For `MERGE_alphagenome_average` (splicing) the cut-points are 0.6727, 0.5339,
0.3926 and 0.2523. These name score ranges; they are not ACMG/AMP classifications or
calibrated probabilities, and the scores are research predictions, not clinical results.

## Reproducibility notes

The features were extracted for these candidate sets with the same checkpoints, scripts
and parameters as the benchmark. GENERATOR and GENERATOR-v2 are float32 and equal the
benchmark values exactly. Evo 2 and Carbon-3B (bfloat16) and Enformer (TensorFlow GPU
kernels) are not bit-deterministic, so recomputing a variant can change those features
slightly.

For variants that are also in the benchmark tables (330 coding, 4,804 splicing), scoring
the benchmark's own feature values with the same models gives a MERGE score in the same
0.2-wide band for 99.4% of coding and 97.4% of splicing variants (median absolute
difference 0.0002 and 0.0013); the differences come from the Evo 2, Carbon-3B and
Enformer features.

## Licence and terms

Released under CC BY-NC 4.0. The files contain outputs of upstream resources that keep
their own terms: AlphaGenome (API offered for non-commercial use), dbNSFP (academic
licence; AlphaMissense and ESM-1b columns), and Nucleotide Transformer (CC BY-NC-SA 4.0).

## Citation

Please cite the manuscript and the MERGE-benchmark repository (`CITATION.cff` there).
