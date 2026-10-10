#!/usr/bin/env python3
"""Step 3 — the ClinVar component of the splicing dataset.

Provenance
----------
This is the original working script (a second file also called ``step1.py``, distinct from
the coding one in ``01_build_coding_dataset.py``) tidied into a CLI. Logic unchanged.

It reads the ClinVar VCF as text — no pysam, no ANNOVAR — and selects splice variants from
ClinVar's **own** ``MC`` molecular-consequence field:

* ``splice_acceptor_variant``
* ``splice_donor_variant``
* ``splice_region_variant``

That last term matters. ANNOVAR's default rule calls a variant ``splicing`` only within
2 bp of an exon–intron boundary; the Sequence Ontology ``splice_region_variant`` reaches
1–3 bp into the exon and 3–8 bp into the intron. Selecting on ``MC`` therefore yields a
substantially larger and differently-shaped set than an ANNOVAR ``splicing`` filter would,
and the two must not be used interchangeably.

Filters
-------
* single-nucleotide REF and ALT; multi-allelic ALT rows are dropped
* ``MC`` contains one of the three splice terms above
* review status of at least two stars — tested as a **substring of the whole
  ``CLNREVSTAT`` string**, so ``criteria_provided,_multiple_submitters,_no_conflicts``
  matches. (The coding builder tests the field token-by-token through pysam and therefore
  does *not* match that status; see the note in ``01_build_coding_dataset.py``. The two
  scripts really do apply different effective thresholds.)
* ``CLNSIG`` free of ``conflicting``, ``uncertain`` and ``vus``; then ``pathogenic`` in the
  string → 1, ``benign`` → 0, both or neither → dropped. Substring matching, so
  ``Likely_pathogenic`` and ``Pathogenic/Likely_pathogenic`` both count as pathogenic.

No chromosome is excluded and no de-duplication is applied — both faithful to the original.

Usage
-----
    python3 03_build_clinvar_splicing.py \
        --clinvar-vcf ../../00_raw_data/clinvar/clinvar_20251221.vcf.gz \
        --output-csv work/clinvar_splicing_snv_2star.csv
"""

from __future__ import annotations

import argparse
import csv
import gzip
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

SPLICING_KEYWORDS = (
    "splice_acceptor_variant",
    "splice_donor_variant",
    "splice_region_variant",
)

# ClinVar review status of at least two stars.
VALID_REVIEW_STATUSES = (
    "criteria_provided,_multiple_submitters,_no_conflicts",  # 2 stars
    "reviewed_by_expert_panel",  # 3 stars
    "practice_guideline",  # 4 stars
)


def parse_info(info_str: str) -> Dict[str, object]:
    info: Dict[str, object] = {}
    for item in info_str.split(";"):
        if "=" in item:
            key, value = item.split("=", 1)
            info[key] = value
        else:
            info[item] = True
    return info


def is_high_quality_star(rev_stat_str: str) -> bool:
    """Two stars and above.

    Substring test against the whole field, so the comma-containing two-star status
    matches. Excluded by construction: ``criteria_provided,_single_submitter``,
    ``criteria_provided,_conflicting_classifications``, ``no_assertion_*``.
    """
    if not rev_stat_str:
        return False
    s = rev_stat_str.lower()
    return any(valid in s for valid in VALID_REVIEW_STATUSES)


def classify_variant(clnsig_str: str) -> Optional[int]:
    """1 = pathogenic, 0 = benign, None = excluded."""
    if not clnsig_str:
        return None
    sig = clnsig_str.lower()

    if "conflicting" in sig:
        return None
    if "uncertain" in sig or "vus" in sig:
        return None

    is_pathogenic = "pathogenic" in sig
    is_benign = "benign" in sig
    if is_pathogenic and is_benign:
        return None
    if is_pathogenic:
        return 1
    if is_benign:
        return 0
    return None


def extract(input_vcf: Path, output_csv: Path) -> None:
    print(f"Reading {input_vcf} ...")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    matched = positives = negatives = 0
    opener = gzip.open if str(input_vcf).endswith(".gz") else open
    mode = "rt" if str(input_vcf).endswith(".gz") else "r"

    with opener(input_vcf, mode, encoding="utf-8") as vcf, \
            output_csv.open("w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(["chr", "pos", "ref", "alt", "label"])

        for line in vcf:
            if line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 8:
                continue

            ref, alt = cols[3], cols[4]
            if "," in alt or len(ref) != 1 or len(alt) != 1:
                continue

            info = parse_info(cols[7])
            mc_val = info.get("MC", "")
            if not any(k in mc_val for k in SPLICING_KEYWORDS):
                continue
            if not is_high_quality_star(info.get("CLNREVSTAT", "")):
                continue

            label = classify_variant(info.get("CLNSIG", ""))
            if label is None:
                continue

            writer.writerow([cols[0], cols[1], ref, alt, label])
            matched += 1
            positives += label == 1
            negatives += label == 0
            if matched % 2000 == 0:
                print(f"  extracted {matched} (label 1: {positives}, label 0: {negatives}) ...")

    print("-" * 40)
    print(f"Wrote {output_csv}")
    print(f"  total    : {matched}")
    print(f"  label 1  : {positives}")
    print(f"  label 0  : {negatives}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clinvar-vcf", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    args = parser.parse_args(argv)

    if not args.clinvar_vcf.exists():
        print(f"ClinVar VCF not found: {args.clinvar_vcf}", file=sys.stderr)
        return 1
    extract(args.clinvar_vcf, args.output_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
