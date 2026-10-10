#!/usr/bin/env python3
"""Shared ClinVar Non-SNV sequence reconstruction utilities."""

from __future__ import annotations

from dataclasses import dataclass

from pyfaidx import Fasta


@dataclass(frozen=True)
class SequencePair:
    ref_sequence: str
    alt_sequence: str
    chrom_key: str
    window_start_1based: int
    window_end_1based: int
    edit_position_1based: int
    edit_ref: str
    edit_alt: str


def normalize_chrom(value: object) -> str:
    text = str(value).strip()
    return text if text.startswith("chr") else f"chr{text}"


def normalize_allele(value: object) -> str:
    return str(value).strip().upper()


def build_chrom_aliases(genome: Fasta) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for key in genome.keys():
        primary = key.split()[0]
        candidates = {key, primary}
        if primary.startswith("chr"):
            candidates.add(primary[3:])
        else:
            candidates.add(f"chr{primary}")
        for candidate in candidates:
            aliases[candidate] = key
    return aliases


def resolve_chrom(chrom: object, aliases: dict[str, str]) -> str:
    text = str(chrom).strip()
    candidates = [text, normalize_chrom(text)]
    if text.startswith("chr"):
        candidates.append(text[3:])
    for candidate in candidates:
        if candidate in aliases:
            return aliases[candidate]
    raise ValueError(f"chromosome not found in FASTA: {chrom}")


def trim_common_prefix(ref: str, alt: str) -> tuple[int, str, str]:
    shift = 0
    while ref and alt and ref[0] == alt[0]:
        ref = ref[1:]
        alt = alt[1:]
        shift += 1
    return shift, ref, alt


def make_sequence_pair(
    genome: Fasta,
    aliases: dict[str, str],
    chrom: object,
    pos: object,
    ref: object,
    alt: object,
    window_bp: int,
) -> SequencePair:
    """Build REF/ALT haplotypes from one shared genomic context.

    The VCF anchor/common left prefix is removed only to identify the true edit
    coordinate. The returned haplotypes retain the same upstream and downstream
    genomic context and may differ in total length for indels.
    """
    ref = normalize_allele(ref)
    alt = normalize_allele(alt)
    if not ref or not alt or any(base not in "ACGT" for base in ref + alt):
        raise ValueError("REF/ALT must be non-empty A/C/G/T strings")
    if len(ref) == 1 and len(alt) == 1:
        raise ValueError("SNV is not supported by this Non-SNV pipeline")

    chrom_key = resolve_chrom(chrom, aliases)
    pos0 = int(pos) - 1
    chrom_len = len(genome[chrom_key])
    if pos0 < 0 or pos0 + len(ref) > chrom_len:
        raise ValueError("variant outside chromosome bounds")

    observed = str(genome[chrom_key][pos0 : pos0 + len(ref)]).upper()
    if observed != ref:
        raise ValueError(f"reference mismatch: FASTA={observed}, input={ref}")

    prefix_len, edit_ref, edit_alt = trim_common_prefix(ref, alt)
    edit_pos0 = pos0 + prefix_len
    if not edit_ref and not edit_alt:
        raise ValueError("REF and ALT are identical after normalization")

    half = window_bp // 2
    start0 = max(0, edit_pos0 - half)
    end0 = min(chrom_len, start0 + window_bp)
    start0 = max(0, end0 - window_bp)
    ref_sequence = str(genome[chrom_key][start0:end0]).upper()
    if any(base not in "ACGT" for base in ref_sequence):
        raise ValueError("sequence window contains non-ACGT bases")

    offset = edit_pos0 - start0
    if offset < 0 or offset + len(edit_ref) > len(ref_sequence):
        raise ValueError("edit falls outside sequence window")
    if edit_ref and ref_sequence[offset : offset + len(edit_ref)] != edit_ref:
        raise ValueError("trimmed REF does not match FASTA")

    alt_sequence = (
        ref_sequence[:offset]
        + edit_alt
        + ref_sequence[offset + len(edit_ref) :]
    )
    return SequencePair(
        ref_sequence=ref_sequence,
        alt_sequence=alt_sequence,
        chrom_key=chrom_key,
        window_start_1based=start0 + 1,
        window_end_1based=end0,
        edit_position_1based=edit_pos0 + 1,
        edit_ref=edit_ref,
        edit_alt=edit_alt,
    )
