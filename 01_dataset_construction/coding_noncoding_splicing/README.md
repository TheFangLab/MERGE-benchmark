# coding / non-coding / splicing dataset construction

The scripts that built `03_datasets/coding_features.csv`, `noncoding_features.csv` and
`splicing_features.csv` — before any model score was attached.

## Provenance

The work was done as a set of separate scripts and shell notes in 2026-05/06. They are
reproduced here with their logic intact; what changed is packaging (argparse instead of
hard-coded paths, chunked reads for the multi-GB tables, logging).

| File | Comes from |
|---|---|
| `00_annotate_clinvar_snv.sh` | working note `annovar注释脚本.txt` |
| `01_build_coding_dataset.py` | working script `step1.py` (the coding one, pysam-based) |
| `02_build_noncoding_dataset.py` | working script `clean_data.py` |
| `03_build_clinvar_splicing.py` | working script `step1.py` (the splicing one, plain-text VCF) |
| `04_gnomad_splice_negatives.sh` | working note `gnomAD阴性数据补充.txt` |
| `05_clean_splicevardb.py` | working script `clean.py` |
| `06_assemble_splicing_dataset.py` | **reconstructed** — the final concatenation was interactive |
| `07_strict_deduplicate_splicing.py` | the strict de-duplication that produced the released splicing table; counts in `splicing_strict_dedup_audit.csv` |

The committed CSVs in `03_datasets/` remain the authoritative dataset definition. Every
filter is also covered by `test_smoke.py`, which runs on synthetic records in a second.

## Run order

```bash
export ANNOVAR_DIR=/path/to/annovar ANNOVAR_DB=/path/to/annovar/humandb
export SPLICEVARDB_RAW=/path/to/all_splicevardb_raw_data.csv
export GNOMAD_DIR=../../00_raw_data/gnomad          # optional, hundreds of GB
WORKDIR=./work bash run_all.sh
```

```
00  ClinVar VCF ──bcftools view -v snps──> SNV-only VCF ──table_annovar refGene──> *_multianno.txt
                                                                       │
01  ClinVar VCF ──pysam, MC=missense──────────────────> coding_snv.csv │  (no ANNOVAR)
                                                                       │
02  *_multianno.txt ──non-coding categories, splicing removed────────> noncoding_snv.csv
03  ClinVar VCF ──text, MC=splice_{acceptor,donor,region}──> clinvar_splicing_snv_2star.csv ─┐
04  gnomAD VCFs ──AF>0.001──> avinput ──table_annovar──> gnomad_result.*_multianno.csv ──────┤
05  SpliceVarDB export ──strict cleaning──> Exonic / Intronic strict CSVs ───────────────────┤
06  concatenate, no de-duplication ─────────────────────────────────────────────────────────┴─> splicing_snv.csv
07  remove every row of every duplicated (chr, pos, ref, alt) key ──────────────────────────────> 35,015 variants
```

Note that steps 01 and 03 both read the ClinVar VCF directly and neither uses the ANNOVAR
output; only the non-coding task does.

Expected output sizes, all confirmed against `03_datasets/`: coding 4,289 (2,912 / 1,377),
non-coding 86,829 (1,340 / 85,489), splicing 35,015 (22,247 / 12,768). Step 06 assembles 35,774 rows (22,848 / 12,926) =
23,413 SpliceVarDB + 10,602 ClinVar + 1,759 gnomAD; step 07
(`07_strict_deduplicate_splicing.py`) then removes every variant whose (chr, pos, ref, alt)
occurs more than once — 379 keys, 759 rows — giving the released table.

## Filters

**Shared label vocabulary.** Pathogenic and Likely pathogenic → 1; Benign and Likely
benign → 0. Uncertain_significance and every `Conflicting…` status are dropped, as is any
record mixing the two directions. Mitochondrial variants are excluded throughout because
the AlphaGenome interface cannot score chrM.

**Coding** is built straight from the ClinVar VCF and never sees ANNOVAR: a variant is
missense if ClinVar's own `MC` field says `missense_variant`, and a substitution if
`CLNVC == single_nucleotide_variant`.

**Non-coding** takes ANNOVAR `Func.refGene` ∈ {`intronic`, `UTR3`, `UTR5`, `upstream`,
`downstream`, `intergenic`} and drops anything mentioning `splicing` in either
`Func.refGene` or `GeneDetail.refGene`. Note that this is an exact-match list: `ncRNA_*`
categories are **not** in it and therefore not in the released non-coding dataset.

**Splicing** is assembled from three independent sources and does **not** come out of the
ANNOVAR table at all:

* **SpliceVarDB** exonic and intronic strict subsets — experimentally validated
  (23,413 rows);
* **ClinVar** variants whose own `MC` field carries `splice_acceptor_variant`,
  `splice_donor_variant` or `splice_region_variant`, at ≥ 2 stars (10,602 rows). The
  `splice_region_variant` term reaches 1–3 bp into the exon and 3–8 bp into the intron, so
  this set is wider than ANNOVAR's 2 bp `splicing` category and the two are not
  interchangeable;
* **gnomAD** variants reaching AF > 0.001 inside the 2 bp splice window, as
  frequency-based proxy negatives — not experimentally confirmed (1,759 rows).

The three are concatenated without de-duplication.

## Tests

```bash
python3 test_smoke.py        # needs pysam + pandas; no downloads
```
