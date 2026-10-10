#!/usr/bin/env python3
"""Build a high-confidence ClinVar Non-SNV dataset.

The script has two stages:
  1. pre: read a bcftools TSV extract, filter/tag variants, write CSV + ANNOVAR avinput.
  2. finalize: read ANNOVAR variant_function output, remove/isolate splice-boundary variants,
     and optionally de-duplicate against existing datasets.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


LOGGER = logging.getLogger("clinvar_non_snv")

PATHOGENIC_TERMS = {"pathogenic", "likely_pathogenic"}
BENIGN_TERMS = {"benign", "likely_benign"}
CONFLICT_TERMS = {
    "conflicting_interpretations_of_pathogenicity",
    "conflicting_classifications_of_pathogenicity",
    "conflicting_data_from_submitters",
}
HIGH_CONF_REVIEW_TERMS = {
    "criteria_provided,_multiple_submitters,_no_conflicts",
    "reviewed_by_expert_panel",
    "practice_guideline",
}


@dataclass(frozen=True)
class Variant:
    chrom: str
    pos: int
    vid: str
    ref: str
    alt: str
    allele_id: str
    clnsig: str
    clnrevstat: str
    clnvc: str
    clnvcso: str
    geneinfo: str
    mc: str
    clndn: str
    label: int
    variant_type: str
    variant_length: int
    av_start: int
    av_end: int
    av_ref: str
    av_alt: str

    @property
    def key(self) -> Tuple[str, int, str, str]:
        return (normalize_chrom(self.chrom), self.pos, self.ref, self.alt)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def split_info_values(value: str) -> Set[str]:
    if value in {"", "."}:
        return set()
    tokens: Set[str] = set()
    for piece in value.replace("|", ",").replace("/", ",").replace(";", ",").split(","):
        token = piece.strip().lower()
        if token:
            tokens.add(token)
    return tokens


def normalize_chrom(chrom: str) -> str:
    return chrom[3:] if chrom.lower().startswith("chr") else chrom


def annovar_chrom(chrom: str) -> str:
    chrom = normalize_chrom(chrom)
    return chrom if chrom.startswith("chr") else f"chr{chrom}"


def normalize_clnsig(raw: str) -> Set[str]:
    terms = split_info_values(raw)
    normalized: Set[str] = set()
    for term in terms:
        normalized.add(term.replace(" ", "_"))
    return normalized


def is_high_conf_review(raw: str) -> bool:
    normalized = raw.strip().lower()
    if normalized in HIGH_CONF_REVIEW_TERMS:
        return True
    terms = split_info_values(raw)
    return bool(terms & HIGH_CONF_REVIEW_TERMS)


def has_conflict(clnsig: str, clnrevstat: str) -> bool:
    clnsig_terms = normalize_clnsig(clnsig)
    review_terms = split_info_values(clnrevstat)
    return bool((clnsig_terms | review_terms) & CONFLICT_TERMS)


def label_from_clnsig(clnsig: str) -> Optional[int]:
    terms = normalize_clnsig(clnsig)
    has_path = bool(terms & PATHOGENIC_TERMS)
    has_benign = bool(terms & BENIGN_TERMS)
    if has_path and not has_benign:
        return 1
    if has_benign and not has_path:
        return 0
    return None


def classify_non_snv(ref: str, alt: str, clnvc: str) -> Optional[Tuple[str, int]]:
    if alt in {"", ".", "*"}:
        return None
    if "," in alt:
        return None

    ref_len = len(ref)
    alt_len = len(alt)
    clnvc_l = clnvc.lower()

    if ref_len == 1 and alt_len == 1:
        return None
    if alt_len > ref_len:
        return ("Insertion", alt_len - ref_len)
    if ref_len > alt_len:
        return ("Deletion", ref_len - alt_len)
    if ref_len == alt_len and ref_len > 1:
        return ("MNV", ref_len)

    if "insertion" in clnvc_l:
        return ("Insertion", max(1, alt_len - ref_len))
    if "deletion" in clnvc_l:
        return ("Deletion", max(1, ref_len - alt_len))
    if "mnv" in clnvc_l or "multiple_nucleotide" in clnvc_l:
        return ("MNV", max(ref_len, alt_len))
    return None


def to_annovar_coords(chrom: str, pos: int, ref: str, alt: str) -> Tuple[int, int, str, str]:
    ref_len = len(ref)
    alt_len = len(alt)
    if alt_len > ref_len:
        if alt.startswith(ref):
            inserted = alt[ref_len:]
            return pos, pos, "-", inserted
        return pos, pos + ref_len - 1, ref, alt
    if ref_len > alt_len:
        if ref.startswith(alt):
            deleted = ref[alt_len:]
            start = pos + alt_len
            end = pos + ref_len - 1
            return start, end, deleted, "-"
        return pos, pos + ref_len - 1, ref, alt
    return pos, pos + ref_len - 1, ref, alt


def parse_tsv(path: Path, max_len: int) -> List[Variant]:
    variants: List[Variant] = []
    counters: Dict[str, int] = {}

    def bump(name: str) -> None:
        counters[name] = counters.get(name, 0) + 1

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"CHROM", "POS", "ID", "REF", "ALT", "CLNSIG", "CLNREVSTAT"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required TSV columns: {', '.join(sorted(missing))}")

        for idx, row in enumerate(reader, start=1):
            if idx % 100000 == 0:
                LOGGER.info("Parsed %s rows; retained %s candidates", idx, len(variants))

            try:
                chrom = row["CHROM"]
                pos = int(row["POS"])
                ref = row["REF"].upper()
                alt = row["ALT"].upper()
            except Exception as exc:
                bump("bad_basic_fields")
                LOGGER.warning("Skipping malformed row %s: %s", idx, exc)
                continue

            non_snv = classify_non_snv(ref, alt, row.get("CLNVC", ""))
            if non_snv is None:
                bump("not_non_snv")
                continue
            variant_type, variant_len = non_snv
            if variant_len > max_len:
                bump("too_long")
                continue
            if has_conflict(row.get("CLNSIG", ""), row.get("CLNREVSTAT", "")):
                bump("conflict")
                continue
            if not is_high_conf_review(row.get("CLNREVSTAT", "")):
                bump("low_review_status")
                continue
            label = label_from_clnsig(row.get("CLNSIG", ""))
            if label is None:
                bump("not_binary_label")
                continue

            av_start, av_end, av_ref, av_alt = to_annovar_coords(chrom, pos, ref, alt)
            variants.append(
                Variant(
                    chrom=normalize_chrom(chrom),
                    pos=pos,
                    vid=row.get("ID", "."),
                    ref=ref,
                    alt=alt,
                    allele_id=row.get("ALLELEID", "."),
                    clnsig=row.get("CLNSIG", "."),
                    clnrevstat=row.get("CLNREVSTAT", "."),
                    clnvc=row.get("CLNVC", "."),
                    clnvcso=row.get("CLNVCSO", "."),
                    geneinfo=row.get("GENEINFO", "."),
                    mc=row.get("MC", "."),
                    clndn=row.get("CLNDN", "."),
                    label=label,
                    variant_type=variant_type,
                    variant_length=variant_len,
                    av_start=av_start,
                    av_end=av_end,
                    av_ref=av_ref,
                    av_alt=av_alt,
                )
            )

    LOGGER.info("Pre-filter counters: %s", counters)
    LOGGER.info("Retained %s high-confidence labeled Non-SNV records", len(variants))
    return variants


def write_pre_outputs(variants: Sequence[Variant], csv_path: Path, avinput_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    avinput_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "chr",
        "pos",
        "ref",
        "alt",
        "label",
        "variant_type",
        "variant_length",
        "clinvar_id",
        "allele_id",
        "clnsig",
        "clnrevstat",
        "clnvc",
        "clnvcso",
        "geneinfo",
        "mc",
        "clndn",
        "annovar_chr",
        "annovar_start",
        "annovar_end",
        "annovar_ref",
        "annovar_alt",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for v in variants:
            writer.writerow(
                {
                    "chr": v.chrom,
                    "pos": v.pos,
                    "ref": v.ref,
                    "alt": v.alt,
                    "label": v.label,
                    "variant_type": v.variant_type,
                    "variant_length": v.variant_length,
                    "clinvar_id": v.vid,
                    "allele_id": v.allele_id,
                    "clnsig": v.clnsig,
                    "clnrevstat": v.clnrevstat,
                    "clnvc": v.clnvc,
                    "clnvcso": v.clnvcso,
                    "geneinfo": v.geneinfo,
                    "mc": v.mc,
                    "clndn": v.clndn,
                    "annovar_chr": annovar_chrom(v.chrom),
                    "annovar_start": v.av_start,
                    "annovar_end": v.av_end,
                    "annovar_ref": v.av_ref,
                    "annovar_alt": v.av_alt,
                }
            )

    with avinput_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        for v in variants:
            writer.writerow(
                [
                    annovar_chrom(v.chrom),
                    v.av_start,
                    v.av_end,
                    v.av_ref,
                    v.av_alt,
                    v.chrom,
                    v.pos,
                    v.ref,
                    v.alt,
                    v.label,
                    v.vid,
                    v.allele_id,
                ]
            )

    LOGGER.info("Wrote pre-ANNOVAR CSV: %s", csv_path)
    LOGGER.info("Wrote ANNOVAR avinput: %s", avinput_path)


def read_existing_keys(paths: Sequence[Path]) -> Set[Tuple[str, int, str, str]]:
    keys: Set[Tuple[str, int, str, str]] = set()
    for path in paths:
        if not path.exists():
            LOGGER.warning("Dedup file does not exist, skipping: %s", path)
            continue
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"chr", "pos", "ref", "alt"}
            if not required.issubset(reader.fieldnames or []):
                LOGGER.warning("Dedup file lacks chr,pos,ref,alt columns, skipping: %s", path)
                continue
            before = len(keys)
            for row in reader:
                try:
                    keys.add((normalize_chrom(row["chr"]), int(row["pos"]), row["ref"].upper(), row["alt"].upper()))
                except Exception:
                    continue
            LOGGER.info("Loaded %s keys from %s", len(keys) - before, path)
    return keys


def load_annovar_functions(path: Path) -> Dict[Tuple[str, int, str, str], Tuple[str, str]]:
    annotations: Dict[Tuple[str, int, str, str], Tuple[str, str]] = {}
    with path.open(newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for idx, row in enumerate(reader, start=1):
            if len(row) < 11:
                LOGGER.warning("Skipping short ANNOVAR row %s: %s", idx, row)
                continue
            func = row[0]
            gene = row[1]
            chrom = normalize_chrom(row[7])
            try:
                pos = int(row[8])
            except ValueError:
                LOGGER.warning("Skipping ANNOVAR row with bad original POS at row %s", idx)
                continue
            ref = row[9].upper()
            alt = row[10].upper()
            annotations[(chrom, pos, ref, alt)] = (func, gene)
    LOGGER.info("Loaded %s ANNOVAR gene annotations from %s", len(annotations), path)
    return annotations


def finalize_dataset(
    pre_csv: Path,
    annovar_variant_function: Path,
    output_csv: Path,
    isolated_splice_csv: Path,
    dedup_paths: Sequence[Path],
) -> None:
    annotations = load_annovar_functions(annovar_variant_function)
    dedup_keys = read_existing_keys(dedup_paths)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    isolated_splice_csv.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    spliced = 0
    deduped = 0
    missing_ann = 0

    with pre_csv.open(newline="") as in_handle, output_csv.open("w", newline="") as out_handle, isolated_splice_csv.open(
        "w", newline=""
    ) as splice_handle:
        reader = csv.DictReader(in_handle)
        base_fields = list(reader.fieldnames or [])
        fieldnames = base_fields + ["annovar_func", "annovar_gene", "is_splice_boundary"]
        out_writer = csv.DictWriter(out_handle, fieldnames=fieldnames)
        splice_writer = csv.DictWriter(splice_handle, fieldnames=fieldnames)
        out_writer.writeheader()
        splice_writer.writeheader()

        for idx, row in enumerate(reader, start=1):
            if idx % 100000 == 0:
                LOGGER.info("Finalized %s rows; kept=%s splice_isolated=%s deduped=%s", idx, kept, spliced, deduped)
            try:
                key = (normalize_chrom(row["chr"]), int(row["pos"]), row["ref"].upper(), row["alt"].upper())
            except Exception as exc:
                LOGGER.warning("Skipping malformed pre-CSV row %s: %s", idx, exc)
                continue
            func, gene = annotations.get(key, (".", "."))
            if func == ".":
                missing_ann += 1
            is_splice = "splicing" in func.lower()
            row.update({"annovar_func": func, "annovar_gene": gene, "is_splice_boundary": int(is_splice)})

            if is_splice:
                splice_writer.writerow(row)
                spliced += 1
                continue
            if key in dedup_keys:
                deduped += 1
                continue
            out_writer.writerow(row)
            kept += 1

    LOGGER.info("Wrote final Non-SNV dataset: %s", output_csv)
    LOGGER.info("Wrote isolated splice-boundary Non-SNV records: %s", isolated_splice_csv)
    LOGGER.info("Summary: kept=%s splice_isolated=%s deduped=%s missing_annovar=%s", kept, spliced, deduped, missing_ann)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    pre = subparsers.add_parser("pre", help="Filter bcftools TSV and write pre-ANNOVAR outputs")
    pre.add_argument("--input-tsv", required=True, type=Path)
    pre.add_argument("--output-csv", required=True, type=Path)
    pre.add_argument("--output-avinput", required=True, type=Path)
    pre.add_argument("--max-len", type=int, default=50)

    fin = subparsers.add_parser("finalize", help="Apply ANNOVAR splice priority and deduplicate")
    fin.add_argument("--pre-csv", required=True, type=Path)
    fin.add_argument("--annovar-variant-function", required=True, type=Path)
    fin.add_argument("--output-csv", required=True, type=Path)
    fin.add_argument("--isolated-splice-csv", required=True, type=Path)
    fin.add_argument("--dedup-csv", type=Path, action="append", default=[])
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "pre":
            variants = parse_tsv(args.input_tsv, args.max_len)
            write_pre_outputs(variants, args.output_csv, args.output_avinput)
        elif args.command == "finalize":
            finalize_dataset(
                pre_csv=args.pre_csv,
                annovar_variant_function=args.annovar_variant_function,
                output_csv=args.output_csv,
                isolated_splice_csv=args.isolated_splice_csv,
                dedup_paths=args.dedup_csv,
            )
        else:
            parser.error(f"Unknown command: {args.command}")
    except Exception as exc:
        LOGGER.exception("Pipeline failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
