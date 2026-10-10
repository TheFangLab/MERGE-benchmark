# 00_raw_data — public source data

Every input of this benchmark comes from a public resource. The large files are **not**
stored in this repository (GitHub caps a single file at 100 MB, and the reference genome
alone is 3.2 GB); `download_raw_data.sh` fetches them with the exact versions used, and
`checksums.md5` records the checksums of the copies that were actually used.

Run everything from this directory:

```bash
bash download_raw_data.sh            # everything except gnomAD
bash download_raw_data.sh gnomad     # gnomAD v4.1 genome sites (very large, opt-in)
md5sum -c checksums.md5              # verify what was downloaded
```

## Variant label sources

| Resource | Version / release | Used for | Download |
|---|---|---|---|
| ClinVar VCF (GRCh38) | `clinvar_20251221.vcf.gz`, 21 December 2025 | positive/negative labels for all four tasks | NCBI FTP `pub/clinvar/vcf_GRCh38/` |
| gnomAD | v4.1 genome sites | common-variant negatives for the splicing task (AF > 0.001, within 2 bp of an exon–intron boundary) | gnomAD public bucket |
| SpliceVarDB | full export, 52,201 rows | experimentally validated splice-altering / normal labels; 23,413 survive the strict cleaning (9,024 exonic + 14,389 intronic) | https://splicevardb.org/ |

ClinVar is the only file whose checksum was recorded at the time of use
(`0e9b0df5bc2f82334df20a3c61dd2494`). gnomAD and SpliceVarDB were consumed as derived
extracts that were not retained; verify those downloads against the checksums published
by the providers.

> **SpliceVarDB has no stable direct download URL.** The script prints the portal link
> and expects you to place the exported table at
> `splicevardb/all_splicevardb_raw_data.csv`; `05_clean_splicevardb.py` derives the strict
> exonic and intronic subsets from it.

## Reference genome and annotation

| Resource | Version | Used for |
|---|---|---|
| hg38 reference FASTA | UCSC `hg38.fa` (+ `.fai`) | sequence windows and haplotype reconstruction for every DNA model |
| GENCODE | release 44 | coding-sequence intervals for the precomputed-score compartment |
| ANNOVAR + refGene | hg38 `refGene` humandb | consequence annotation with a 2 bp splice boundary |

ANNOVAR requires a (free) academic registration and cannot be downloaded
non-interactively; the script prints the registration URL. Its `annotate_variation.pl`
and `humandb/` path are passed to the construction scripts through `ANNOVAR_DIR` and
`ANNOVAR_DB`.

## Precomputed variant-effect scores

These are official releases consumed by exact `(chr, pos, ref, alt)` lookup — no model
was retrained or fine-tuned.

| Score | Source |
|---|---|
| AlphaMissense (hg38) and ESM-1b | dbNSFP 5.3a, columns `AlphaMissense_score` and `ESM1b_score`, matched on chromosome, position, REF and ALT (https://www.dbnsfp.org/releases/). The dbNSFP copy is the one used; AlphaMissense's own release differs slightly from it. |
| GPN-MSA (hg38, genome-wide) | Hugging Face dataset `songlab/gpn-msa-hg38-scores` |

## Model checkpoints

Fetched from Hugging Face at run time by the extraction scripts in
`02_feature_extraction/`. Exact identifiers:

```
LongSafari/hyenadna-medium-160k-seqlen-hf
InstaDeepAI/nucleotide-transformer-500m-human-ref
InstaDeepAI/nucleotide-transformer-v2-500m-multi-species
arcinstitute/evo2_7b
arcinstitute/evo2_7b_base
arcinstitute/evo2_1b_base
HuggingFaceBio/Carbon-3B
GenerTeam/GENERator-eukaryote-1.2b-base
GenerTeam/GENERator-v2-eukaryote-1.2b-base
```

Enformer was run from the TensorFlow Hub model (`deepmind/enformer/1`) with
`02_feature_extraction/shared/enformer_summary_shard.py`. AlphaGenome is not a
downloadable checkpoint: variant effects were obtained through the official
`alphagenome.models.dna_client` API, which requires an API key.

## What is stored in the repository

Nothing in this directory except this README, the download script and the checksum file.
The derived, analysis-ready tables live in `03_datasets/` and are small enough to be
version controlled.
