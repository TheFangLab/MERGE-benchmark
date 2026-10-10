#!/usr/bin/env python3
"""Step 2 — the non-coding dataset, from the ANNOVAR-annotated ClinVar SNV table
produced by ``00_annotate_clinvar_snv.sh``.

Provenance
----------
This is the original working script (``clean_data.py``): the same ``CLNSIG=`` /
``CLNREVSTAT=`` regex pull out of the ANNOVAR ``Otherinfo`` passthrough, the same
review-status whitelist, the same ``Func.refGene`` category list, the same splicing
exclusion, and the same explicit Pathogenic / Benign label vocabulary.


The splicing task's ClinVar component is *not* the complement of this filter and is not
built here: splice variants are selected from ClinVar's own ``MC`` field, which includes
``splice_region_variant`` and so reaches well beyond ANNOVAR's 2 bp window. See
``03_build_clinvar_splicing.py``.

Two implementation notes, both behaviour-preserving:

* the table is read in chunks (default 200,000 rows) instead of in one go — the annotated
  ClinVar SNV table is several GB;
* the ``Otherinfo`` column carrying the VCF ``INFO`` string is located by scanning a sample
  of rows rather than the whole file.

Usage
-----
    python3 02_build_noncoding_dataset.py \
        --annotated work/clinvar_annotated.hg38_multianno.txt \
        --output-csv work/noncoding_snv.csv

Expected: 86,829 variants (1,340 pathogenic / 85,489 benign).
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

import pandas as pd

LOGGER = logging.getLogger("partition_clinvar")

# ClinVar review status of at least two stars.
HIGH_QUALITY_STATUSES = [
    "criteria_provided,_multiple_submitters,_no_conflicts",
    "reviewed_by_expert_panel",
    "practice_guideline",
]

# ANNOVAR Func.refGene categories treated as non-coding.
NON_CODING_REGIONS = ["intronic", "UTR3", "UTR5", "upstream", "downstream", "intergenic"]

# Explicit label vocabulary — anything else (including Uncertain_significance and every
# Conflicting… term) is dropped rather than guessed at.
TP_LABELS = ["Pathogenic", "Likely_pathogenic", "Pathogenic/Likely_pathogenic"]
TN_LABELS = ["Benign", "Likely_benign", "Benign/Likely_benign"]

OUT_COLUMNS = ["chr", "pos", "ref", "alt", "label", "Func.refGene", "Gene.refGene", "CLNSIG"]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def extract_vcf_info(row_str, key: str) -> Optional[str]:
    if pd.isna(row_str):
        return None
    match = re.search(rf"{key}=([^;]+)", str(row_str))
    return match.group(1) if match else None


def find_info_column(path: Path, sample_rows: int = 20000) -> str:
    """Locate the ANNOVAR Otherinfo column that carries the VCF INFO string."""
    sample = pd.read_csv(path, sep="\t", nrows=sample_rows, low_memory=False)
    for col in sample.columns:
        if sample[col].astype(str).str.contains("CLNSIG=").any():
            LOGGER.info("INFO column: %s", col)
            return col
    raise ValueError(
        f"No column containing 'CLNSIG=' in the first {sample_rows} rows of {path}. "
        "Was the annotation run with -vcfinput?"
    )


def iter_chunks(path: Path, chunksize: int) -> Iterator[pd.DataFrame]:
    yield from pd.read_csv(path, sep="\t", low_memory=False, chunksize=chunksize)


def normalize_chrom(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"^chr", "", regex=True)


def filter_chunk(chunk: pd.DataFrame, info_col: str) -> pd.DataFrame:
    chunk = chunk.copy()
    chunk["CLNSIG"] = chunk[info_col].apply(lambda x: extract_vcf_info(x, "CLNSIG"))
    chunk["CLNREVSTAT"] = chunk[info_col].apply(lambda x: extract_vcf_info(x, "CLNREVSTAT"))

    hq = chunk[chunk["CLNREVSTAT"].isin(HIGH_QUALITY_STATUSES)]
    if hq.empty:
        return hq

    func = hq["Func.refGene"].astype(str)
    detail = hq["GeneDetail.refGene"].astype(str)
    selected = hq[
        func.isin(NON_CODING_REGIONS)
        & (~func.str.contains("splicing", na=False))
        & (~detail.str.contains("splicing", na=False))
    ]
    if selected.empty:
        return selected

    selected = selected.copy()
    selected["label"] = pd.NA
    selected.loc[selected["CLNSIG"].isin(TP_LABELS), "label"] = 1
    selected.loc[selected["CLNSIG"].isin(TN_LABELS), "label"] = 0
    selected = selected.dropna(subset=["label"])
    if selected.empty:
        return selected

    selected["chr"] = normalize_chrom(selected["Chr"])
    selected["pos"] = selected["Start"].astype(int)
    selected["ref"] = selected["Ref"].astype(str).str.upper()
    selected["alt"] = selected["Alt"].astype(str).str.upper()
    selected["label"] = selected["label"].astype(int)
    return selected[OUT_COLUMNS]


def run(annotated: Path, output_csv: Path, chunksize: int, exclude_chroms: Sequence[str]) -> None:
    info_col = find_info_column(annotated)
    excluded = {c.strip().upper() for c in exclude_chroms if c.strip()}

    kept: List[pd.DataFrame] = []
    seen_rows = 0
    for chunk in iter_chunks(annotated, chunksize):
        seen_rows += len(chunk)
        part = filter_chunk(chunk, info_col)
        if not part.empty:
            kept.append(part)
        if seen_rows % (chunksize * 10) == 0:
            LOGGER.info("  read %s rows; kept %s", seen_rows, sum(len(p) for p in kept))

    if not kept:
        raise SystemExit("No non-coding variants survived the filters — check the input table.")

    result = pd.concat(kept, ignore_index=True)
    result = result[~result["chr"].str.upper().isin(excluded)]
    result = result.drop_duplicates(subset=["chr", "pos", "ref", "alt"], keep="first")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)

    positives = int((result["label"] == 1).sum())
    LOGGER.info("Read %s annotated rows", seen_rows)
    LOGGER.info(
        "Wrote %s — %s non-coding variants (%s pathogenic / %s benign)",
        output_csv, len(result), positives, len(result) - positives,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotated", required=True, type=Path,
                        help="ANNOVAR *_multianno.txt from step 0")
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--chunksize", type=int, default=200000)
    parser.add_argument(
        "--exclude-chrom",
        default="MT,M",
        help="Comma-separated chromosomes to drop (default: MT,M — AlphaGenome cannot score chrM)",
    )
    args = parser.parse_args(argv)

    try:
        run(args.annotated, args.output_csv, args.chunksize, args.exclude_chrom.split(","))
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Non-coding build failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
