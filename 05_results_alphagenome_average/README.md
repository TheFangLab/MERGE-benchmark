# 05_results_alphagenome_average — fixed MERGE + AlphaGenome average

```
merge_alphagenome_average = 0.5 × MERGE score + 0.5 × AlphaGenome percentile
```

Offered for splicing and noncoding variants only. Nothing is fitted: the weight is
fixed at 1:1 and the AlphaGenome half is a mid-rank percentile (the fraction of
reference variants scoring below, plus half of those tied). The AlphaGenome column is
fixed by variant type: `alphagenome_splicing` for splicing and
`alphagenome_raw_score_max` for noncoding variants.

**Evaluation (out of fold).** Missing AlphaGenome scores (571 splicing variants) take
the training-fold median, in the training and held-out folds alike; each held-out
variant is then converted to its percentile against the training folds. The
AlphaGenome component is therefore identical to the standalone AlphaGenome score in
Table 1, and every variant of the fixed cohort is scored.

**Deployment.** The web server and `merge-cli` compute the average only for variants
with an AlphaGenome score, using `alphagenome_reference_quantiles.json`: every distinct
AlphaGenome score among the benchmark variants of that type with its mid-rank
percentile, interpolated linearly (no labels used).

## Files

| File | Content |
|---|---|
| `{splicing,noncoding}_alphagenome_average_oof.csv` | Per variant: label, fold, the average, the MERGE score, the AlphaGenome percentile, the raw AlphaGenome value, and whether it was observed |
| `{splicing,noncoding}_alphagenome_average_meta.json` | Definition and missing-value policy |
| `alphagenome_average_summary.csv` | Fold mean ± sample SD for the average, MERGE and the AlphaGenome component |
| `alphagenome_reference_quantiles.json` | The deployed percentile table |
| `merge_final_thresholds.json` | Interpretation cut-points (the file name is the one the tools load) |

## Results (Table 2)

| Variant type | n | Average | MERGE | AlphaGenome |
|---|---:|---:|---:|---:|
| Splicing | 35,015 | 0.9567 / 0.9740 | 0.9269 / 0.9561 | 0.9563 / 0.9710 |
| Noncoding | 86,829 | 0.9781 / 0.7702 | 0.9472 / 0.6841 | 0.9774 / 0.7953 |

AUROC / AUPRC, mean over the three chromosome-held-out folds. On noncoding variants
the AlphaGenome raw maximum keeps the higher AUPRC.

## Interpretation cut-points

The average is not on the scale of the MERGE score, so it has its own cut-points. For
each MERGE cut (0.8, 0.6, 0.4, 0.2) the corresponding cut is the quantile of the
average at the fraction of variants that MERGE places below it, computed on the
out-of-fold predictions of variants with an observed AlphaGenome score. No labels
are used.

| Variant type | Cut-points (high → low) | Band agreement with MERGE |
|---|---|---:|
| Splicing | 0.6727 / 0.5339 / 0.3926 / 0.2523 | 0.812 |
| Noncoding | 0.8150 / 0.6827 / 0.5764 / 0.4975 | 0.876 |

These bands describe score ranges; they are not calibrated clinical probabilities.

## Licence

`alphagenome_reference_quantiles.json` summarizes AlphaGenome output, whose API is
offered for non-commercial use, and is released under CC BY-NC 4.0.
