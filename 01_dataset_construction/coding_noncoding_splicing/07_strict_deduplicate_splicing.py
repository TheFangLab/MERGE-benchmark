#!/usr/bin/env python3
"""
Step 07: strict deduplication of the assembled splicing table.

`06_assemble_splicing_dataset.py` concatenates SpliceVarDB, ClinVar and gnomAD
without collapsing overlaps, so a variant present in two sources appears twice.
That gives 35,774 rows but only 35,394 distinct (chr, pos, ref, alt) keys.

This step removes **every row of every key that occurs more than once** — 379
keys, 759 rows — rather than keeping one copy or averaging. A duplicated variant
either carries conflicting labels from different sources or carries feature
values that were scored from different source records; neither can be resolved
from the released table, which has no source column, so no copy is kept. The
result is the released `03_datasets/splicing_features.csv`: 35,015 variants,
22,247 positive and 12,768 negative, with no duplicate keys. Row order is the
order of the assembled table.
(The released table then has its GPN_MSA_score column re-matched to the
official GPN-MSA table, 02_feature_extraction/shared/match_gpn_msa.py; see
03_datasets/README.md.)

A more lenient rule (drop conflicting keys, average same-label duplicates) gives
35,383 rows; it is not the release.

Usage:
    python 07_strict_deduplicate_splicing.py <assembled.csv> <out.csv> [audit.csv]
"""
import sys

import pandas as pd

KEY = ["chr", "pos", "ref", "alt"]


def deduplicate(df):
    dup = df.duplicated(KEY, keep=False)
    out = df.loc[~dup].reset_index(drop=True)
    audit = {
        "rows_before": len(df),
        "unique_variant_keys_before": int(df.drop_duplicates(KEY).shape[0]),
        "duplicate_variant_keys": int(df.loc[dup].drop_duplicates(KEY).shape[0]),
        "rows_in_duplicate_keys_removed": int(dup.sum()),
        "rows_after": len(out),
        "positive_after": int((out["label"] == 1).sum()),
        "negative_after": int((out["label"] == 0).sum()),
        "duplicate_keys_after": int(out.duplicated(KEY).sum()),
    }
    return out, audit


def main():
    src, dst = sys.argv[1], sys.argv[2]
    df = pd.read_csv(src, low_memory=False, dtype={"chr": str})
    out, audit = deduplicate(df)
    out.to_csv(dst, index=False)
    if len(sys.argv) > 3:
        pd.DataFrame(list(audit.items()), columns=["metric", "value"]).to_csv(sys.argv[3], index=False)
    for k, v in audit.items():
        print(f"{k:34s} {v:,}")


if __name__ == "__main__":
    main()
