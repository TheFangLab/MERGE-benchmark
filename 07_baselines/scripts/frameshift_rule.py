"""The rule-based control for the non-SNV dataset.

On an 81 %-positive ClinVar indel set, "does the length change break the reading
frame" is already a substantial classifier, so it belongs in the comparison: it
puts a number on how much of the task is explained by that one fact, and it
answers the objection that an ensemble on this dataset has only learned to
detect frameshifts.

The score is the indicator itself (1 for a length change that is not a multiple
of three, 0 otherwise), so its AUROC is (sensitivity + specificity) / 2. This is
deliberately the crudest possible form of the rule -- no gene model is consulted,
so it does not know whether the variant is even in coding sequence.

Usage: python frameshift_rule.py <features.csv> <out-dir> [task]
"""
import os
import sys

import pandas as pd


def main():
    features, outdir = sys.argv[1], sys.argv[2]
    task = sys.argv[3] if len(sys.argv) > 3 else "nonsnv"
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(features, usecols=["chr", "pos", "ref", "alt", "label"],
                     low_memory=False)
    df["chr"] = df["chr"].astype(str).str.replace("chr", "", regex=False)
    delta = (df["ref"].astype(str).str.len() - df["alt"].astype(str).str.len()).abs()
    df["score"] = ((delta % 3) != 0).astype(int)

    tp = int(((df.score == 1) & (df.label == 1)).sum())
    fp = int(((df.score == 1) & (df.label == 0)).sum())
    fn = int(((df.score == 0) & (df.label == 1)).sum())
    tn = int(((df.score == 0) & (df.label == 0)).sum())
    sens, spec = tp / (tp + fn), tn / (tn + fp)
    print(f"[{task}] frameshift rule: sensitivity {sens:.3f}, specificity "
          f"{spec:.3f}, AUROC {(sens + spec) / 2:.4f}")

    out = os.path.join(outdir, f"{task}__frameshift_rule.tsv")
    df[["chr", "pos", "ref", "alt", "score"]].drop_duplicates(
        subset=["chr", "pos", "ref", "alt"]).to_csv(out, sep="\t", index=False)
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
