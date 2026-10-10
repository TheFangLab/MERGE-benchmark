"""
Build the BRCA1 SGE external cohort from Findlay et al. (2018), Supplementary Table 1.

Keeps missense SNVs classified LOF or FUNC that ClinVar did not list at the source
time, lifts them from hg19 to hg38 (UCSC chain, chr17 blocks) and checks the hg38
reference base. Presence in the 21 December 2025 ClinVar release is recorded as a
flag (every one of these variants is now in ClinVar, so the strict ClinVar-absent
set is empty); the MERGE validation cohort is the 1,577 variants absent from the
coding training table. The cohort is written as CSV and VCF together with a
per-step flow table (cohort/sge_cohort_flow.csv).

External inputs (not redistributed): the hg19-to-hg38 chain, hg38 FASTA with .fai,
and the ClinVar VCF -- the same files listed in 00_raw_data/.

Usage:
  python 01_build_sge_cohort.py --chain hg19ToHg38.over.chain.gz --fasta hg38.fa \
      --clinvar clinvar_20251221.vcf.gz [--out ../cohort]
"""
from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import pandas as pd

EXP = Path(__file__).resolve().parent.parent
REPO = EXP.parent
SOURCE = EXP / "source" / "GSE117159_Supplementary_Table_1.xlsx"
CODING_TRAIN = REPO / "03_datasets" / "coding_features.csv"
OUT = CHAIN = FASTA = FAI = CLINVAR = None      # set from the command line in main()


def complement(base: str) -> str:
    return base.translate(str.maketrans("ACGT", "TGCA"))


def load_chain_blocks(path: Path, source_chrom: str):
    blocks = []
    active = None
    t_pos = q_pos = 0
    with gzip.open(path, "rt") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                active = None
                continue
            if line.startswith("chain "):
                f = line.split()
                active = {
                    "score": int(f[1]), "t_name": f[2], "t_strand": f[4],
                    "t_start": int(f[5]), "q_name": f[7], "q_size": int(f[8]),
                    "q_strand": f[9], "q_start": int(f[10]), "chain_id": f[12],
                }
                t_pos = active["t_start"]
                q_pos = active["q_start"]
                # UCSC hg19ToHg38 chain stores hg19 on the target (t) side
                # and hg38 on the query (q) side. We therefore invert the
                # usual query-to-target lookup for our hg19 source position.
                if active["t_name"] != source_chrom:
                    active = None
                continue
            if active is None:
                continue
            nums = list(map(int, line.split()))
            size = nums[0]
            blocks.append({**active, "q_block_start": q_pos, "q_block_end": q_pos + size,
                           "t_block_start": t_pos, "t_block_end": t_pos + size})
            if len(nums) == 3:
                t_pos += size + nums[1]
                q_pos += size + nums[2]
    return sorted(blocks, key=lambda b: (-b["score"], b["q_block_start"]))


def liftover_snv(pos1: int, blocks):
    t0 = pos1 - 1
    hits = []
    for b in blocks:
        if b["t_block_start"] <= t0 < b["t_block_end"]:
            offset = t0 - b["t_block_start"]
            q_oriented = b["q_block_start"] + offset
            if b["q_strand"] == "+":
                q0 = q_oriented
            else:
                q0 = b["q_size"] - q_oriented - 1
            hits.append((b["score"], b["q_name"], q0 + 1, b["q_strand"], b["chain_id"]))
    if not hits:
        return None
    hits.sort(reverse=True)
    best = hits[0]
    if len(hits) > 1 and hits[1][0] == best[0] and hits[1][1:3] != best[1:3]:
        return None
    return best[1:]


def read_fai(path: Path):
    out = {}
    with path.open() as handle:
        for line in handle:
            name, length, offset, line_bases, line_width, *_ = line.rstrip().split("\t")
            out[name] = tuple(map(int, (length, offset, line_bases, line_width)))
    return out


def fasta_base(handle, fai, chrom: str, pos1: int) -> str:
    length, offset, line_bases, line_width = fai[chrom]
    if not 1 <= pos1 <= length:
        return ""
    p0 = pos1 - 1
    byte = offset + (p0 // line_bases) * line_width + (p0 % line_bases)
    handle.seek(byte)
    return handle.read(1).decode().upper()


def clinvar_keys(path: Path, targets):
    positions = {(c, p) for c, p, _, _ in targets}
    hits = set()
    with gzip.open(path, "rt", errors="replace") as handle:
        for line in handle:
            if not line or line[0] == "#":
                continue
            f = line.rstrip().split("\t", 8)
            chrom = f[0].removeprefix("chr")
            try:
                pos = int(f[1])
            except ValueError:
                continue
            if (chrom, pos) not in positions:
                continue
            ref = f[3].upper()
            for alt in f[4].upper().split(","):
                key = (chrom, pos, ref, alt)
                if key in targets:
                    hits.add(key)
    return hits


def main():
    global OUT, CHAIN, FASTA, FAI, CLINVAR
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", required=True, type=Path)
    ap.add_argument("--fasta", required=True, type=Path)
    ap.add_argument("--clinvar", required=True, type=Path)
    ap.add_argument("--out", default=EXP / "cohort", type=Path)
    a = ap.parse_args()
    OUT, CHAIN, FASTA, CLINVAR = a.out, a.chain, a.fasta, a.clinvar
    FAI = Path(str(FASTA) + ".fai")
    OUT.mkdir(exist_ok=True)
    raw = pd.read_excel(SOURCE, header=3)
    raw.columns = [str(c).strip() for c in raw.columns]
    raw.insert(0, "source_row", range(5, len(raw) + 5))
    report = [{"stage": "official_table", "n": len(raw)}]

    cohort = raw[
        raw["consequence"].astype(str).str.contains("missense", case=False, na=False)
        & raw["func.class"].isin(["LOF", "FUNC"])
        & raw["clinvar_simple"].astype(str).str.strip().str.lower().eq("absent")
    ].copy()
    cohort["label"] = cohort["func.class"].map({"LOF": 1, "FUNC": 0}).astype(int)
    report.append({"stage": "missense_LOF_FUNC_source_ClinVar_absent", "n": len(cohort)})

    cohort["chr_hg19"] = cohort["chromosome"].astype(str).str.removeprefix("chr")
    cohort["pos_hg19"] = pd.to_numeric(cohort["position (hg19)"], errors="coerce").astype("Int64")
    cohort["ref_hg19"] = cohort["reference"].astype(str).str.upper()
    cohort["alt_hg19"] = cohort["alt"].astype(str).str.upper()
    valid = cohort["pos_hg19"].notna() & cohort["ref_hg19"].isin(list("ACGT")) & cohort["alt_hg19"].isin(list("ACGT"))
    cohort = cohort[valid].copy()
    report.append({"stage": "valid_hg19_snv", "n": len(cohort)})

    blocks = load_chain_blocks(CHAIN, "chr17")
    mapped = [liftover_snv(int(p), blocks) for p in cohort["pos_hg19"]]
    cohort["liftover"] = mapped
    cohort["chr"] = [x[0].removeprefix("chr") if x else pd.NA for x in mapped]
    cohort["pos"] = [x[1] if x else pd.NA for x in mapped]
    cohort["strand"] = [x[2] if x else pd.NA for x in mapped]
    cohort["chain_id"] = [x[3] if x else pd.NA for x in mapped]
    cohort = cohort[cohort["liftover"].notna()].copy()
    cohort["ref"] = [complement(r) if s == "-" else r for r, s in zip(cohort["ref_hg19"], cohort["strand"])]
    cohort["alt"] = [complement(a) if s == "-" else a for a, s in zip(cohort["alt_hg19"], cohort["strand"])]
    report.append({"stage": "unique_hg38_liftover", "n": len(cohort)})

    fai = read_fai(FAI)
    with FASTA.open("rb") as fasta:
        cohort["hg38_reference_base"] = [fasta_base(fasta, fai, "chr" + str(c), int(p)) for c, p in zip(cohort["chr"], cohort["pos"])]
    cohort["ref_match_hg38"] = cohort["ref"] == cohort["hg38_reference_base"]
    cohort = cohort[cohort["ref_match_hg38"]].copy()
    report.append({"stage": "hg38_reference_match", "n": len(cohort)})

    key_cols = ["chr", "pos", "ref", "alt"]
    dup = cohort.duplicated(key_cols, keep=False)
    cohort[dup].to_csv(OUT / "sge_liftover_duplicate_rows.csv", index=False)
    cohort = cohort.drop_duplicates(key_cols, keep="first")
    report.append({"stage": "unique_hg38_variant", "n": len(cohort)})

    targets = set(map(tuple, cohort[key_cols].itertuples(index=False, name=None)))
    cv_hits = clinvar_keys(CLINVAR, targets)
    cohort["present_in_local_ClinVar_20251221"] = [tuple(x) in cv_hits for x in cohort[key_cols].itertuples(index=False, name=None)]
    report.append({
        "stage": "absent_from_local_ClinVar_20251221",
        "n": int((~cohort["present_in_local_ClinVar_20251221"].astype(bool)).sum()),
    })

    train = pd.read_csv(CODING_TRAIN, usecols=key_cols, low_memory=False)
    train["chr"] = train["chr"].astype(str).str.removeprefix("chr")
    train["pos"] = pd.to_numeric(train["pos"], errors="coerce").astype("Int64")
    train["ref"] = train["ref"].astype(str).str.upper()
    train["alt"] = train["alt"].astype(str).str.upper()
    train_keys = set(map(tuple, train[key_cols].dropna().itertuples(index=False, name=None)))
    cohort["overlaps_original_coding_training"] = [tuple(x) in train_keys for x in cohort[key_cols].itertuples(index=False, name=None)]
    report.append({
        "stage": "absent_from_original_coding_training",
        "n": int((~cohort["overlaps_original_coding_training"].astype(bool)).sum()),
    })
    cohort["strict_external_eligible"] = (
        ~cohort["present_in_local_ClinVar_20251221"].astype(bool)
        & ~cohort["overlaps_original_coding_training"].astype(bool)
    )
    # The labels remain independent SGE experimental labels even though all
    # variants were subsequently deposited in ClinVar. For evaluating MERGE,
    # the operative leakage exclusion is direct overlap with its actual coding
    # training table, not later database presence alone.
    cohort["merge_external_eligible"] = ~cohort["overlaps_original_coding_training"].astype(bool)

    cohort = cohort.sort_values(["chr", "pos", "ref", "alt"]).reset_index(drop=True)
    cohort.insert(0, "variant_id", [f"SGE_{i:04d}" for i in range(1, len(cohort) + 1)])
    cohort.to_csv(OUT / "sge_source_time_clinvar_absent_hg38.csv", index=False)
    cohort[cohort["strict_external_eligible"]].to_csv(
        OUT / "sge_strict_external_eligible_hg38.csv", index=False
    )
    external = cohort[cohort["merge_external_eligible"]].copy()
    external.to_csv(OUT / "sge_merge_external_validation_hg38.csv", index=False)
    pd.DataFrame(report).to_csv(OUT / "sge_cohort_flow.csv", index=False)

    with (OUT / "sge_source_time_clinvar_absent_hg38.vcf").open("w", newline="\n") as out:
        out.write("##fileformat=VCFv4.2\n##reference=GRCh38\n")
        out.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for _, row in cohort.iterrows():
            out.write(
                f"chr{row['chr']}\t{int(row['pos'])}\t{row['variant_id']}\t"
                f"{row['ref']}\t{row['alt']}\t.\tPASS\t"
                f"SGE_LABEL={int(row['label'])};FUNC_CLASS={row['func.class']}\n"
            )

    with (OUT / "sge_merge_external_validation_hg38.vcf").open("w", newline="\n") as out:
        out.write("##fileformat=VCFv4.2\n##reference=GRCh38\n")
        out.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for _, row in external.iterrows():
            out.write(
                f"chr{row['chr']}\t{int(row['pos'])}\t{row['variant_id']}\t"
                f"{row['ref']}\t{row['alt']}\t.\tPASS\t"
                f"SGE_LABEL={int(row['label'])};FUNC_CLASS={row['func.class']}\n"
            )

    print(pd.DataFrame(report).to_string(index=False))
    print("labels", cohort["label"].value_counts().sort_index().to_dict())
    print("final", len(cohort))
    print("merge_external", len(external), external["label"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
