#!/usr/bin/env python3
"""Step 6 — assemble the splicing SNV dataset from its three sources.

The three sources and their label conventions are the original ones; the concatenation
itself was an interactive pandas step, so this file is a reconstruction of that last step
only.

1. **SpliceVarDB** exonic and intronic strict subsets, from ``05_clean_splicevardb.py`` —
   experimentally validated: splice-altering -> 1, normal -> 0.  **23,413 rows**
   (9,024 exonic + 14,389 intronic).
2. **ClinVar** splice SNVs selected on the ``MC`` field, >= 2 stars, from
   ``03_build_clinvar_splicing.py``.  Pathogenic -> 1, benign -> 0.  10,607 rows, of which
   **10,602** survive the ACGT check below.
3. **gnomAD v4.1** SNVs with AF > 0.001 inside ANNOVAR's 2 bp splice window, from
   ``04_gnomad_splice_negatives.sh`` — negatives (label 0), frequency-based proxy controls.
   **1,759 rows.**

23,413 + 10,602 + 1,759 = **35,774**, the row count of the assembled table.
``07_strict_deduplicate_splicing.py`` then removes every duplicated variant to give the
released 35,015-row ``03_datasets/splicing_features.csv``.

The sources are concatenated as they are: no de-duplication, and no particular source
order.

Five ClinVar records carry ``ALT = "."`` — a missing alternate allele that survives a
``len(alt) == 1`` test. They are dropped here by the explicit ACGT check, which is what
brings the ClinVar contribution from 10,607 to the 10,602 rows the released table holds.

Mitochondrial variants are dropped (AlphaGenome cannot score chrM).

Usage
-----
    python3 06_assemble_splicing_dataset.py \
        --splicevardb work/SpliceVarDB_Exonic_SNV_Strict.csv \
        --splicevardb work/SpliceVarDB_Intronic_SNV_Strict.csv \
        --clinvar work/clinvar_splicing_snv_2star.csv \
        --gnomad-multianno work/gnomad_result.hg38_multianno.csv \
        --output-csv work/splicing_snv.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

LOGGER = logging.getLogger("splicing_dataset")

MITO_CHROMS = {"m", "mt", "chrm", "chrmt"}
OUT_FIELDS = ["chr", "pos", "ref", "alt", "label", "source", "evidence"]

# Column aliases, so a re-export of any source with slightly different headers still works.
CHR_ALIASES = ("chr", "chrom", "chromosome")
POS_ALIASES = ("pos", "pos_hg38", "position", "start", "hg38_pos")
REF_ALIASES = ("ref", "reference", "ref_allele")
ALT_ALIASES = ("alt", "alternate", "alt_allele")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def normalize_chrom(chrom: str) -> str:
    chrom = str(chrom).strip()
    return chrom[3:] if chrom.lower().startswith("chr") else chrom


def is_snv(ref: str, alt: str) -> bool:
    return len(ref) == 1 and len(alt) == 1 and ref in "ACGT" and alt in "ACGT" and ref != alt


def pick(fieldnames: Sequence[str], aliases: Sequence[str], path: Path) -> str:
    lowered = {name.lower(): name for name in fieldnames}
    for alias in aliases:
        if alias in lowered:
            return lowered[alias]
    raise ValueError(f"None of {aliases} found in {path}; columns = {list(fieldnames)}")


def emit(rows: List[Dict[str, object]], chrom, pos, ref, alt, label: int, source: str, evidence: str) -> bool:
    chrom = normalize_chrom(chrom)
    if chrom.lower() in MITO_CHROMS:
        return False
    ref = str(ref).strip().upper()
    alt = str(alt).strip().upper()
    if not is_snv(ref, alt):
        return False
    try:
        position = int(float(str(pos).strip()))
    except ValueError:
        return False
    rows.append(
        {"chr": chrom, "pos": position, "ref": ref, "alt": alt, "label": label, "source": source, "evidence": evidence}
    )
    return True


def open_table(path: Path):
    delim = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    handle = path.open(newline="")
    return handle, csv.DictReader(handle, delimiter=delim)


# ------------------------------------------------------------------------------------
def read_clinvar(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    handle, reader = open_table(path)
    with handle:
        fields = reader.fieldnames or []
        c, p, r, a = (
            pick(fields, CHR_ALIASES, path),
            pick(fields, POS_ALIASES, path),
            pick(fields, REF_ALIASES, path),
            pick(fields, ALT_ALIASES, path),
        )
        for row in reader:
            emit(rows, row[c], row[p], row[r], row[a], int(row["label"]), "clinvar",
                 row.get("Func.refGene", "MC splice term"))
    LOGGER.info("ClinVar splice SNVs kept: %s", len(rows))
    return rows


def read_gnomad(path: Path, af_threshold: float) -> List[Dict[str, object]]:
    """ANNOVAR multianno output of the gnomAD avinput; keep splice-region rows only.

    The avinput carried AF as its single passthrough column, which ANNOVAR echoes as
    ``Otherinfo1``.  If that column is present the AF threshold is re-applied here as a
    safety net; if it is absent the rows are taken as already filtered by step 3.
    """
    rows: List[Dict[str, object]] = []
    handle, reader = open_table(path)
    skipped_non_splice = 0
    with handle:
        fields = reader.fieldnames or []
        c, p, r, a = (
            pick(fields, CHR_ALIASES, path),
            pick(fields, POS_ALIASES, path),
            pick(fields, REF_ALIASES, path),
            pick(fields, ALT_ALIASES, path),
        )
        func_col = next((f for f in fields if f.lower() == "func.refgene"), None)
        if func_col is None:
            raise ValueError(f"No Func.refGene column in {path}; columns = {fields}")
        af_col = next((f for f in fields if f.lower() in {"af", "otherinfo1", "allele_frequency"}), None)

        for row in reader:
            if "splicing" not in str(row[func_col]).lower():
                skipped_non_splice += 1
                continue
            if af_col is not None:
                try:
                    if float(row[af_col]) <= af_threshold:
                        continue
                except (TypeError, ValueError):
                    pass  # non-numeric passthrough: keep, step 3 already filtered
            emit(rows, row[c], row[p], row[r], row[a], 0, "gnomad", f"AF>{af_threshold:g}, {row[func_col]}")
    LOGGER.info("gnomAD common splice-region negatives: %s (skipped %s non-splice rows)", len(rows), skipped_non_splice)
    return rows


def read_splicevardb(paths: Sequence[Path]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for path in paths:
        if not path.exists():
            LOGGER.warning("SpliceVarDB file missing, skipping: %s", path)
            continue
        before = len(rows)
        handle, reader = open_table(path)
        with handle:
            fields = reader.fieldnames or []
            c, p, r, a = (
                pick(fields, CHR_ALIASES, path),
                pick(fields, POS_ALIASES, path),
                pick(fields, REF_ALIASES, path),
                pick(fields, ALT_ALIASES, path),
            )
            for row in reader:
                emit(
                    rows, row[c], row[p], row[r], row[a],
                    int(row["label"]), "splicevardb", row.get("classification", "."),
                )
        LOGGER.info("SpliceVarDB %s: +%s", path.name, len(rows) - before)
    return rows


# ------------------------------------------------------------------------------------
def concatenate(chunks: Sequence[List[Dict[str, object]]], output: Path) -> None:
    """Straight concatenation. Overlapping variants are kept, not collapsed — that is what
    the released table does."""
    seen: Set[Tuple[str, int, str, str]] = set()
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    positives = 0
    repeats = 0
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUT_FIELDS)
        writer.writeheader()
        for chunk in chunks:
            for row in chunk:
                key = (row["chr"], row["pos"], row["ref"], row["alt"])  # type: ignore[index]
                if key in seen:
                    repeats += 1
                else:
                    seen.add(key)
                writer.writerow(row)
                written += 1
                positives += int(row["label"])  # type: ignore[arg-type]
    LOGGER.info("%s rows repeat a key already contributed by another source (kept)", repeats)
    LOGGER.info(
        "Wrote %s — %s rows, %s unique keys (%s splice-altering / %s normal-or-common)",
        output, written, len(seen), positives, written - positives,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clinvar", required=True, type=Path)
    parser.add_argument("--gnomad-multianno", type=Path, default=None,
                        help="omit to assemble the SpliceVarDB + ClinVar part only")
    parser.add_argument("--splicevardb", action="append", required=True, type=Path,
                        help="Repeat for the exonic and intronic subsets")
    parser.add_argument("--af-threshold", type=float, default=0.001)
    parser.add_argument("--output-csv", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        concatenate(
            [
                read_splicevardb(args.splicevardb),
                read_clinvar(args.clinvar),
                read_gnomad(args.gnomad_multianno, args.af_threshold) if args.gnomad_multianno else [],
            ],
            args.output_csv,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Splicing dataset assembly failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
