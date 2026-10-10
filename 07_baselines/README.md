# 07_baselines — comparison with established predictors

Manuscript Tables 7 and 8 and Figures 7 and 8: MERGE, and on splicing and noncoding
variants the fixed MERGE + AlphaGenome average, against published predictors on the
same chromosome-held-out folds.

## Layout

```
scripts/    score extraction, evaluation, significance and Figures 7-8
cadd_raw/   raw output of the CADD v1.7 web service for the benchmark VCFs
scores/     <task>__<tool>.tsv   chr pos ref alt score   (larger = more damaging)
results/    baseline_comparison.csv            every predictor, common subset and own coverage
            baseline_comparison_by_region.csv  noncoding, by ANNOVAR region class
            significance.csv                   Table 8 (chromosome-cluster bootstrap)
```

`scores/` holds one row per scored variant; a variant a tool cannot score is absent and
is never imputed. SIFT4G and PROVEAN are negated so that larger always means more
damaging. The exception is AlphaGenome, whose files carry the benchmark's own
out-of-fold percentile (training-fold median imputation of missing scores, as in the
individual-feature benchmark); for noncoding variants the raw and quantile maxima are
two separate comparators.

## Predictors

| Task | Comparators |
|---|---|
| Coding | CADD, REVEL, ClinPred, MetaRNN, BayesDel, VARITY, VEST4, M-CAP, MutFormer, SIFT4G, PolyPhen-2 HVAR, PROVEAN, MutationAssessor, PrimateAI, GERP++, phyloP 100-way, AlphaGenome |
| Splicing | SpliceAI, Pangolin, CADD, phyloP 100-way, AlphaGenome |
| Noncoding | CADD, LINSIGHT, phyloP 100-way, ncER, ReMM v0.3.1 and v0.4, AlphaGenome raw and quantile maximum |
| Non-SNV | CADD, SpliceAI, Pangolin, a frameshift rule (positive when the length change is not divisible by three), AlphaGenome |

Group B (trained on clinical pathogenicity labels): REVEL, ClinPred, MetaRNN, BayesDel,
VARITY, VEST4, M-CAP, MutFormer, ncER, ReMM. Everything else is group A. Coding scores
other than CADD come from dbNSFP 5.3a; the `*_dbnsfp` files are dbNSFP copies of CADD
and phyloP kept as a cross-check of the direct extraction and are not shown in the
figures.

## Running it

Score extraction needs the external resources named in each script (dbNSFP 5.3a, the
noncoding score tracks, SpliceAI, Pangolin, the CADD web service, hg38, ANNOVAR); none is
redistributed. The comparison itself runs from the committed files:

```bash
# extraction (once; outputs are committed in scores/)
python scripts/make_vcfs.py                          # datasets -> vcf/
bash   scripts/cadd_submit.sh vcf cadd_raw           # CADD web service
python scripts/collect_cadd.py cadd_raw scores
python scripts/extract_dbnsfp.py <dbNSFP5.3a_grch38.gz> <sites.csv> scores coding
python scripts/extract_mutformer_release.py          # MutFormer hg38 release
python scripts/extract_noncoding.py noncoding <sites.csv> <data-dir> scores
python scripts/extract_bigwig.py <phyloP100way.bw> phyloP100way <sites.csv> scores <task>
bash   scripts/prep_splice_vcf.sh vcf/splicing.vcf hg38.fa.fai vcf/splicing.chr.vcf
bash   scripts/run_spliceai.sh vcf/splicing.chr.vcf work/splicing_spliceai hg38.fa 16
bash   scripts/run_pangolin.sh .
python scripts/parse_splice_vcf.py spliceai work/splicing_spliceai.vcf scores/splicing__SpliceAI.tsv
python scripts/frameshift_rule.py ../03_datasets/non_snv_features.csv scores nonsnv
bash   scripts/annotate_regions.sh <sites.csv> <annovar-dir> noncoding

# comparison (minutes on a CPU)
python scripts/extract_alphagenome_feature.py        # scores/<task>__AlphaGenome*.tsv
python scripts/evaluate.py                           # results/baseline_comparison*.csv
python scripts/merge_vs_baseline_ci.py --table8      # results/significance.csv
python scripts/make_baseline_figures.py              # 06_figures/fig7, fig8
```

## How the comparison is computed

* MERGE and the average enter as their out-of-fold predictions from the
  chromosome-held-out folds, aligned to the dataset by row.
* Table 7 and Figures 7–8 use the variants scored by every predictor of the task (the
  common subset: coding 2,876; splicing 33,177; noncoding 85,233; non-SNV 33,157).
  Metrics are computed per outer fold and reported as mean ± sample SD.
  `baseline_comparison.csv` also reports each predictor on its own covered variants.
* Table 8 compares pooled out-of-fold predictions on the variants scored by both
  methods, with 95% percentile intervals from 2,000 resamples of whole chromosomes
  (`numpy.random.default_rng(42)`, one stream per comparison).
