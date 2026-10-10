#!/usr/bin/env python3
"""Step 1 — build the coding missense SNV dataset directly from the ClinVar VCF.

Provenance
----------
This is the original working script (``step1.py``) tidied into a CLI: same filters, same
label logic, same helper functions.  Two things were added and are flagged in ``--help``:

* ``--exclude-chrom`` (default ``MT,M``) — the released ``03_datasets/coding_features.csv``
  contains no mitochondrial variants, because the AlphaGenome interface cannot score chrM.
  The original script dropped them at a later stage; doing it here keeps the dataset
  definition in one place.
* the primary output is one combined CSV.  The four ``*_imbalanced.txt`` files that
  ``step1.py`` wrote (a chromosome 1/3/5 hold-out) are still available via
  ``--legacy-split-dir``; the released benchmark does not use them — the modelling code in
  ``04_modeling/`` does its own chromosome-held-out cross-validation.

Selection criteria
------------------
* single-nucleotide REF and ALT, and ClinVar ``CLNVC == single_nucleotide_variant``
* ClinVar ``MC`` contains ``missense_variant``
* review status: **expert panel (3 stars) or practice guideline (4 stars)** for the
  released dataset; ``--review-status two-star`` widens it to two stars and above
* ``CLNSIG`` free of any ``Conflicting…`` term, and unambiguous: Pathogenic / Likely
  pathogenic (label 1) or Benign / Likely benign (label 0); anything mixed with
  Uncertain_significance, or mixing the two directions, is dropped
* duplicate ``(chr, pos, ref, alt)`` keys collapse to one record

Expected result: 4,289 variants (2,912 pathogenic / 1,377 benign).

Usage
-----
    python3 01_build_coding_dataset.py \
        --clinvar-vcf ../../00_raw_data/clinvar/clinvar_20251221.vcf.gz \
        --output-csv work/coding_snv.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pysam

LOGGER = logging.getLogger("coding_dataset")

# Review statuses accepted, keyed by --review-status.
REVIEW_STATUS_SETS = {
    "as-released": {
        "reviewed_by_expert_panel",  # 3 stars
        "practice_guideline",  # 4 stars
    },
    "two-star": {
        "criteria_provided,_multiple_submitters,_no_conflicts",  # 2 stars
        "reviewed_by_expert_panel",
        "practice_guideline",
    },
}

PATHOGENIC_TERMS = {"pathogenic", "likely_pathogenic"}
BENIGN_TERMS = {"benign", "likely_benign"}
VUS_TERMS = {"uncertain_significance"}

OUT_FIELDS = ["chr", "pos", "ref", "alt", "label", "clnsig", "clnrevstat", "geneinfo", "clinvar_id"]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def normalize_chrom(chrom_name: str) -> str:
    chrom_name = str(chrom_name)
    if chrom_name.startswith("chr"):
        return chrom_name[3:]
    return chrom_name


def as_list(value) -> List[str]:
    """ClinVar INFO fields come back as str, tuple or None depending on the header."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def review_status(record) -> str:
    """Re-join what pysam split on the commas inside CLNREVSTAT."""
    return ",".join(as_list(record.info.get("CLNREVSTAT")))


def check_star_rating(record, accepted: Set[str]) -> bool:
    return review_status(record) in accepted


def is_nssnv(record) -> bool:
    """Single-nucleotide substitution that ClinVar annotates as missense."""
    if len(record.ref) != 1 or len(record.alts[0]) != 1:
        return False

    clnvc = record.info.get("CLNVC")
    if clnvc is not None and clnvc != "single_nucleotide_variant":
        return False

    mc_list = as_list(record.info.get("MC"))
    if not any("missense_variant" in mc for mc in mc_list):
        return False

    return True


def get_clinical_label(record) -> Optional[int]:
    """Binary label, excluding every kind of interpretation conflict.

    Dropped: anything carrying a ``Conflicting…`` term, and any record that mixes
    benign with VUS, pathogenic with VUS, or benign with pathogenic.
    """
    sig_set = set()
    for item in as_list(record.info.get("CLNSIG")):
        if not item:
            continue
        for part in item.split("|"):
            sig_set.add(part.lower())

    for s in sig_set:
        if "conflicting" in s:
            return None

    has_patho = bool(sig_set & PATHOGENIC_TERMS)
    has_benign = bool(sig_set & BENIGN_TERMS)
    has_vus = bool(sig_set & VUS_TERMS)

    if has_benign and has_vus:
        return None
    if has_patho and has_vus:
        return None
    if has_benign and has_patho:
        return None

    if has_patho:
        return 1
    if has_benign:
        return 0
    return None


def process_clinvar(
    vcf_path: Path, exclude_chroms: Sequence[str], accepted_review: Set[str]
) -> Dict[Tuple[str, int, str, str], Dict[str, object]]:
    excluded = {c.strip().upper() for c in exclude_chroms if c.strip()}
    variants: Dict[Tuple[str, int, str, str], Dict[str, object]] = {}
    counters = {"scanned": 0, "not_missense_snv": 0, "low_review": 0, "no_binary_label": 0, "excluded_chrom": 0}

    LOGGER.info("Parsing ClinVar: %s", vcf_path)
    with pysam.VariantFile(str(vcf_path)) as vcf:
        for record in vcf:
            counters["scanned"] += 1
            if counters["scanned"] % 200000 == 0:
                LOGGER.info("  scanned %s records; kept %s", counters["scanned"], len(variants))

            if not record.alts:
                continue
            if not is_nssnv(record):
                counters["not_missense_snv"] += 1
                continue
            if not check_star_rating(record, accepted_review):
                counters["low_review"] += 1
                continue
            label = get_clinical_label(record)
            if label is None:
                counters["no_binary_label"] += 1
                continue

            chrom = normalize_chrom(record.chrom)
            if chrom.upper() in excluded:
                counters["excluded_chrom"] += 1
                continue

            key = (chrom, record.pos, record.ref, record.alts[0])
            variants.setdefault(
                key,
                {
                    "chr": chrom,
                    "pos": record.pos,
                    "ref": record.ref,
                    "alt": record.alts[0],
                    "label": label,
                    "clnsig": "|".join(as_list(record.info.get("CLNSIG"))) or ".",
                    "clnrevstat": review_status(record) or ".",
                    "geneinfo": (record.info.get("GENEINFO") or "."),
                    "clinvar_id": record.id or ".",
                },
            )

    LOGGER.info("Filter counters: %s", counters)
    return variants


def write_csv(rows: Sequence[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    positives = sum(int(r["label"]) for r in rows)
    LOGGER.info("Wrote %s — %s variants (%s pathogenic / %s benign)", path, len(rows), positives, len(rows) - positives)


def write_legacy_split(rows: Sequence[Dict[str, object]], directory: Path, test_chroms: Sequence[str]) -> None:
    """The chromosome 1/3/5 hold-out written by the original step1.py.

    Not used by the released benchmark; 04_modeling/ re-splits by chromosome itself.
    """
    directory.mkdir(parents=True, exist_ok=True)
    test = {c.strip() for c in test_chroms if c.strip()}
    buckets = {
        ("train", 1): [], ("train", 0): [],
        ("test", 1): [], ("test", 0): [],
    }
    for row in rows:
        split = "test" if str(row["chr"]) in test else "train"
        buckets[(split, int(row["label"]))].append(row)

    names = {
        ("train", 1): "train_pos_imbalanced.txt",
        ("train", 0): "train_neg_imbalanced.txt",
        ("test", 1): "test_pos_imbalanced.txt",
        ("test", 0): "test_neg_imbalanced.txt",
    }
    for (split, label), bucket in buckets.items():
        path = directory / names[(split, label)]
        with path.open("w") as handle:
            handle.write("Chrom\tPos\tRef\tAlt\tLabel\n")
            for row in bucket:
                handle.write(f"{row['chr']}\t{row['pos']}\t{row['ref']}\t{row['alt']}\t{label}\n")
        LOGGER.info("Wrote %s (%s rows)", path, len(bucket))


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clinvar-vcf", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument(
        "--exclude-chrom",
        default="MT,M",
        help="Comma-separated chromosomes to drop (default: MT,M — AlphaGenome cannot score chrM)",
    )
    parser.add_argument(
        "--review-status",
        choices=sorted(REVIEW_STATUS_SETS),
        default="as-released",
        help=(
            "as-released (default): expert panel + practice guideline, reproduces the released CSV; "
            "two-star: also accept criteria_provided,_multiple_submitters,_no_conflicts. "
            "See the module docstring."
        ),
    )
    parser.add_argument(
        "--legacy-split-dir",
        type=Path,
        default=None,
        help="Also write the original step1.py chromosome hold-out into this directory",
    )
    parser.add_argument("--test-chroms", default="1,3,5", help="Test chromosomes for --legacy-split-dir")
    args = parser.parse_args(argv)

    if not args.clinvar_vcf.exists():
        LOGGER.error("ClinVar VCF not found: %s", args.clinvar_vcf)
        return 1

    accepted_review = REVIEW_STATUS_SETS[args.review_status]
    LOGGER.info("Review-status filter '%s': %s", args.review_status, sorted(accepted_review))
    variants = process_clinvar(args.clinvar_vcf, args.exclude_chrom.split(","), accepted_review)
    rows = sorted(variants.values(), key=lambda r: (str(r["chr"]), int(r["pos"]), r["ref"], r["alt"]))
    write_csv(rows, args.output_csv)
    if args.legacy_split_dir is not None:
        write_legacy_split(rows, args.legacy_split_dir, args.test_chroms.split(","))
    return 0


if __name__ == "__main__":
    sys.exit(main())
