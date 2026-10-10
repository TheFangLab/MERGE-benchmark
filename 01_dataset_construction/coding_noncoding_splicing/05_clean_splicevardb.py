#!/usr/bin/env python3
"""Step 4 — clean the SpliceVarDB export into strict exonic and intronic SNV subsets.

Provenance
----------
This is the original working script (``clean.py``) tidied into a CLI, with its logic and
its three deliberate corrections intact:

1. **Strict low-confidence exclusion.** Any ``classification`` containing ``low-frequency``,
   ``conflict`` or ``uncertain`` (case-insensitive substring, not exact match) is dropped.
2. **Conflicting variants are removed entirely.** If one ``(chr, pos, ref, alt)`` key
   carries both label 1 and label 0 anywhere in the export, every row for that key goes —
   it is not resolved by majority or by first-seen.
3. **De-duplication only after step 2**, so keeping the first occurrence is safe.

Everything is read as string (``dtype=str``) so that positions are never silently coerced
to floats, which would break de-duplication.

Labels: ``classification`` containing ``Splice-altering`` -> 1, containing ``Normal`` -> 0,
anything else dropped. Only ``location == Exonic`` and ``location == Intronic`` are kept —
the ambiguous "Splice region" tier is left out, matching the EVO2 comparison the splicing
task is benchmarked against.

Reference counts from the original run: 1,181 exonic and 3,769 intronic variants.

Usage
-----
    python3 05_clean_splicevardb.py \
        --input all_splicevardb_raw_data.csv \
        --output-exonic work/SpliceVarDB_Exonic_SNV_Strict.csv \
        --output-intronic work/SpliceVarDB_Intronic_SNV_Strict.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

LOGGER = logging.getLogger("splicevardb")

EXCLUDE_KEYWORDS = ["low-frequency", "conflict", "uncertain"]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_label(classification: str) -> Optional[int]:
    cls = classification.strip()
    if "Splice-altering" in cls:
        return 1
    if "Normal" in cls:
        return 0
    return None


def clean(
    input_file: Path,
    output_exonic: Path,
    output_intronic: Path,
    pos_column: str,
) -> None:
    # Everything as string: a numeric pos column would defeat de-duplication.
    df = pd.read_csv(input_file, dtype=str)
    LOGGER.info("Raw rows: %s", len(df))

    # 1. drop low-confidence / conflicting / uncertain classifications
    pattern = "|".join(EXCLUDE_KEYWORDS)
    df = df.dropna(subset=["classification"])
    df_clean = df[~df["classification"].str.lower().str.contains(pattern)].copy()
    LOGGER.info("After removing low-frequency / conflict / uncertain: %s", len(df_clean))

    # 2. keep single-nucleotide substitutions
    df_clean["ref"] = df_clean["ref"].str.strip().str.upper()
    df_clean["alt"] = df_clean["alt"].str.strip().str.upper()
    df_clean = df_clean[
        (df_clean["ref"].str.len() == 1)
        & (df_clean["alt"].str.len() == 1)
        & (df_clean["ref"] != "-")
        & (df_clean["alt"] != "-")
    ]
    LOGGER.info("After SNV filter: %s", len(df_clean))

    # 3. labels
    df_clean["label"] = df_clean["classification"].apply(get_label)
    df_clean = df_clean.dropna(subset=["label"])
    df_clean["label"] = df_clean["label"].astype(int)

    # 4. remove variants with conflicting labels across submissions
    dedup_cols = ["chr", pos_column, "ref", "alt"]
    conflict_check = df_clean.groupby(dedup_cols)["label"].nunique()
    conflict_variants = conflict_check[conflict_check > 1].index
    is_conflict = df_clean.set_index(dedup_cols).index.isin(conflict_variants)
    LOGGER.info("Conflicting variants removed: %s", len(conflict_variants))
    df_clean = df_clean[~is_conflict]

    # 5. de-duplicate (safe now that conflicts are gone)
    df_clean = df_clean.drop_duplicates(subset=dedup_cols, keep="first")
    LOGGER.info("After conflict resolution and de-duplication: %s", len(df_clean))

    # 6. split by location
    df_clean["location"] = df_clean["location"].str.strip()
    for location, out_path in (("Exonic", output_exonic), ("Intronic", output_intronic)):
        subset = df_clean[df_clean["location"] == location].copy()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        subset.to_csv(out_path, index=False)
        LOGGER.info(
            "%s: %s variants (%s splice-altering / %s normal) -> %s",
            location,
            len(subset),
            int((subset["label"] == 1).sum()),
            int((subset["label"] == 0).sum()),
            out_path,
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, help="SpliceVarDB export (all_splicevardb_raw_data.csv)")
    parser.add_argument("--output-exonic", required=True, type=Path)
    parser.add_argument("--output-intronic", required=True, type=Path)
    parser.add_argument("--pos-column", default="pos_hg38", help="hg38 position column in the export")
    args = parser.parse_args(argv)

    try:
        clean(args.input, args.output_exonic, args.output_intronic, args.pos_column)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("SpliceVarDB cleaning failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
