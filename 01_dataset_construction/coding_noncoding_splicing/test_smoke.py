#!/usr/bin/env python3
"""Smoke-test the coding / non-coding / splicing construction scripts on synthetic input.

This does not touch the real ClinVar / gnomAD / SpliceVarDB data — it builds a handful of
synthetic records that exercise every filter, then checks exactly which ones survive:
review status, missense selection, VUS and conflict exclusion, chrM removal,
de-duplication, ANNOVAR consequence routing, SpliceVarDB conflict handling, the AF
threshold, and cross-source duplicate precedence.

Needs pysam and pandas:

    python3 test_smoke.py
"""
import csv, subprocess, sys, tempfile, textwrap
from pathlib import Path

SRC = Path(__file__).resolve().parent
WORK = Path(tempfile.mkdtemp(prefix="merge_dataset_smoke_"))
PY = sys.executable
print(f"working directory: {WORK}")

def run(*args):
    print("\n$", " ".join(str(a) for a in args))
    r = subprocess.run([str(a) for a in args], capture_output=True, text=True)
    print(r.stdout[-3000:]); print(r.stderr[-3000:])
    assert r.returncode == 0, f"FAILED: {args}"
    return r

# ---------------- 1. coding: synthetic ClinVar VCF -----------------------------------
vcf = WORK / "clinvar_mini.vcf"
header = textwrap.dedent("""\
##fileformat=VCFv4.2
##INFO=<ID=CLNSIG,Number=.,Type=String,Description="sig">
##INFO=<ID=CLNREVSTAT,Number=.,Type=String,Description="rev">
##INFO=<ID=CLNVC,Number=1,Type=String,Description="vc">
##INFO=<ID=MC,Number=.,Type=String,Description="mc">
##INFO=<ID=GENEINFO,Number=1,Type=String,Description="gene">
##contig=<ID=1>
##contig=<ID=3>
##contig=<ID=MT>
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO
""")
def rec(chrom,pos,vid,ref,alt,sig,rev,vc="single_nucleotide_variant",mc="SO:0001583|missense_variant",gene="G:1"):
    return f"{chrom}\t{pos}\t{vid}\t{ref}\t{alt}\t.\t.\tCLNSIG={sig};CLNREVSTAT={rev};CLNVC={vc};MC={mc};GENEINFO={gene}\n"
STAR2 = "criteria_provided,_multiple_submitters,_no_conflicts"
rows = [
    rec("1",100,"1","A","G","Pathogenic",STAR2),                        # keep -> 1
    rec("1",200,"2","C","T","Likely_benign","reviewed_by_expert_panel"),# keep -> 0
    rec("1",201,"2b","C","T","Likely_benign","reviewed_by_expert_panel"),# dup pos differs, keep -> 0
    rec("3",300,"3","G","A","Pathogenic","criteria_provided,_single_submitter"),   # drop: 1 star
    rec("3",400,"4","G","A","Uncertain_significance",STAR2),            # drop: VUS
    rec("3",500,"5","G","A","Conflicting_classifications_of_pathogenicity",STAR2), # drop: conflict
    rec("3",600,"6","G","A","Pathogenic",STAR2,mc="SO:0001819|synonymous_variant"),# drop: not missense
    rec("3",700,"7","GA","G","Pathogenic",STAR2,vc="Deletion"),         # drop: not SNV
    rec("MT",800,"8","T","C","Pathogenic",STAR2),                       # drop: chrM
    rec("1",100,"9","A","G","Pathogenic",STAR2),                        # duplicate key -> collapsed
]
vcf.write_text(header + "".join(rows))
subprocess.run(["bgzip","-f",str(vcf)], check=False)
gz = Path(str(vcf)+".gz")
if not gz.exists():
    import pysam
    pysam.tabix_compress(str(vcf), str(gz), force=True)
# default (as-released): expert panel / practice guideline only
run(PY, SRC/"01_build_coding_dataset.py", "--clinvar-vcf", gz,
    "--output-csv", WORK/"coding_snv.csv", "--legacy-split-dir", WORK/"legacy", "--test-chroms","1,3,5")
coding = list(csv.DictReader((WORK/"coding_snv.csv").open()))
assert sorted((r["chr"], int(r["pos"]), r["label"]) for r in coding) == [("1",200,"0"),("1",201,"0")], coding
print("OK coding as-released:", coding)

# two-star: the 2-star pathogenic record is now accepted, chrM/VUS/conflict/dup still out
run(PY, SRC/"01_build_coding_dataset.py", "--clinvar-vcf", gz,
    "--output-csv", WORK/"coding_2star.csv", "--review-status","two-star")
coding2 = list(csv.DictReader((WORK/"coding_2star.csv").open()))
assert sorted((r["chr"], int(r["pos"]), r["label"]) for r in coding2) == [("1",100,"1"),("1",200,"0"),("1",201,"0")], coding2
print("OK coding two-star:", coding2)
assert (WORK/"legacy"/"test_pos_imbalanced.txt").exists()

# ---------------- 2. noncoding / splicing partition ----------------------------------
ann = WORK / "clinvar_annotated.hg38_multianno.txt"
cols = ["Chr","Start","End","Ref","Alt","Func.refGene","Gene.refGene","GeneDetail.refGene",
        "ExonicFunc.refGene","AAChange.refGene","Otherinfo1","Otherinfo2","Otherinfo3"]
def arow(chrom,start,ref,alt,func,detail,sig,rev=STAR2,gene="G"):
    return [chrom,start,start,ref,alt,func,gene,detail,".",".","0.5","PASS",
            f"ALLELEID=1;CLNSIG={sig};CLNREVSTAT={rev};CLNVC=single_nucleotide_variant"]
arows = [
    arow("1","1000","A","G","intronic",".","Benign"),                       # noncoding 0
    arow("1","1100","A","G","UTR5",".","Pathogenic"),                       # noncoding 1
    arow("1","1200","A","G","intergenic",".","Benign/Likely_benign"),       # noncoding 0
    arow("1","1300","A","G","intronic",".","Uncertain_significance"),       # drop
    arow("1","1400","A","G","intronic",".","Benign","criteria_provided,_single_submitter"), # drop 1 star
    arow("1","1500","A","G","intronic","NM_1:splicing","Benign"),           # drop: splicing in GeneDetail
    arow("1","1600","A","G","exonic",".","Pathogenic"),                     # drop: coding
    arow("2","1700","A","G","splicing",".","Pathogenic"),                   # splicing 1
    arow("2","1800","A","G","exonic;splicing",".","Likely_benign"),         # splicing 0
    arow("MT","1900","A","G","intronic",".","Benign"),                      # drop chrM
    arow("1","1000","A","G","intronic",".","Benign"),                       # duplicate
]
with ann.open("w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t"); w.writerow(cols); w.writerows(arows)

run(PY, SRC/"02_build_noncoding_dataset.py","--annotated",ann,
    "--output-csv",WORK/"noncoding_snv.csv","--chunksize","3")
nc = list(csv.DictReader((WORK/"noncoding_snv.csv").open()))
assert len(nc)==3, nc
assert sorted((r["pos"],r["label"]) for r in nc)==[("1000","0"),("1100","1"),("1200","0")], nc
print("OK noncoding:", nc)

# ---------------- 2b. ClinVar splicing component (MC field, not ANNOVAR) --------------
svcf = WORK / "clinvar_splice.vcf"
srows = [
    rec("2",1700,"s1","A","G","Pathogenic",STAR2,mc="SO:0001574|splice_acceptor_variant"),
    rec("2",1800,"s2","A","G","Likely_benign",STAR2,mc="SO:0001630|splice_region_variant"),
    rec("2",1900,"s3","A","G","Pathogenic",STAR2,mc="SO:0001575|splice_donor_variant"),
    rec("2",2000,"s4","A","G","Pathogenic",STAR2,mc="SO:0001583|missense_variant"),
    rec("2",2100,"s5","A","G","Pathogenic","criteria_provided,_single_submitter",
        mc="SO:0001630|splice_region_variant"),
    rec("2",2200,"s6","A","G","Uncertain_significance",STAR2,mc="SO:0001630|splice_region_variant"),
    rec("2",2300,"s7","A","G","Conflicting_classifications_of_pathogenicity",STAR2,
        mc="SO:0001630|splice_region_variant"),
    rec("2",2400,"s8","AT","A","Pathogenic",STAR2,mc="SO:0001630|splice_region_variant"),
]
svcf.write_text(header + "".join(srows))
subprocess.run(["bgzip","-f",str(svcf)], check=False)
sgz = Path(str(svcf)+".gz")
if not sgz.exists():
    import pysam; pysam.tabix_compress(str(svcf), str(sgz), force=True)
run(PY, SRC/"03_build_clinvar_splicing.py","--clinvar-vcf",sgz,
    "--output-csv",WORK/"clinvar_splicing.csv")
sp = list(csv.DictReader((WORK/"clinvar_splicing.csv").open()))
assert len(sp)==3, sp
assert sorted((r["pos"],r["label"]) for r in sp)==[("1700","1"),("1800","0"),("1900","1")], sp
print("OK clinvar splicing (MC):", sp)

# ---------------- 3. SpliceVarDB cleaning --------------------------------------------
svdb = WORK/"splicevardb_raw.csv"
with svdb.open("w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["chr","pos_hg38","ref","alt","classification","location"])
    w.writerows([
        ["1","5000","A","G","Splice-altering (high confidence)","Exonic"],   # 1
        ["1","5100","C","T","Normal","Exonic"],                              # 0
        ["1","5200","G","A","Low-frequency splice-altering","Exonic"],       # drop
        ["1","5300","G","A","Conflicting","Intronic"],                       # drop
        ["1","5400","G","A","Splice-altering","Intronic"],                   # 1
        ["1","5400","G","A","Normal","Intronic"],                            # conflict -> both dropped
        ["1","5500","GA","G","Normal","Intronic"],                           # drop: not SNV
        ["1","5600","T","C","Normal","Splice region"],                       # dropped by location
        ["1","5700","T","C","Normal","Intronic"],                            # 0
        ["1","5700","T","C","Normal","Intronic"],                            # dup -> keep first
    ])
run(PY, SRC/"05_clean_splicevardb.py","--input",svdb,
    "--output-exonic",WORK/"svdb_exonic.csv","--output-intronic",WORK/"svdb_intronic.csv")
ex = list(csv.DictReader((WORK/"svdb_exonic.csv").open()))
intr = list(csv.DictReader((WORK/"svdb_intronic.csv").open()))
assert len(ex)==2 and len(intr)==1, (ex,intr)
assert intr[0]["pos_hg38"]=="5700", intr
print("OK splicevardb:", ex, intr)

# ---------------- 4. gnomAD multianno + assembly --------------------------------------
gn = WORK/"gnomad_result.hg38_multianno.csv"
with gn.open("w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["Chr","Start","End","Ref","Alt","Func.refGene","Gene.refGene","GeneDetail.refGene",
                "ExonicFunc.refGene","AAChange.refGene","Otherinfo1"])
    w.writerows([
        ["chr2","1700","1700","A","G","splicing","G",".",".",".","0.02"],  # dup of clinvar -> dropped
        ["chr4","9000","9000","A","G","splicing","G",".",".",".","0.02"],  # keep 0
        ["chr4","9100","9100","A","G","intronic","G",".",".",".","0.02"],  # drop: not splicing
        ["chr4","9200","9200","A","G","splicing","G",".",".",".","0.0001"],# drop: AF too low
        ["chrM","9300","9300","A","G","splicing","G",".",".",".","0.02"],  # drop chrM
    ])
run(PY, SRC/"06_assemble_splicing_dataset.py","--clinvar",WORK/"clinvar_splicing.csv",
    "--gnomad-multianno",gn,"--splicevardb",WORK/"svdb_exonic.csv","--splicevardb",WORK/"svdb_intronic.csv",
    "--output-csv",WORK/"splicing_snv.csv")
final = list(csv.DictReader((WORK/"splicing_snv.csv").open()))
srcs = [r["source"] for r in final]
# concatenation, not de-duplication: a gnomAD row repeating a ClinVar key is kept
assert len(final)==3+3+2, final
assert srcs.count("splicevardb")==3 and srcs.count("clinvar")==3 and srcs.count("gnomad")==2, srcs
keys = [(r["chr"], r["pos"], r["ref"], r["alt"]) for r in final]
assert len(keys) - len(set(keys)) == 1, "the overlapping ClinVar/gnomAD key must survive twice"
print("OK splicing assembly (no de-duplication):", final)

# ---------------- 5. strict de-duplication --------------------------------------------
run(PY, SRC/"07_strict_deduplicate_splicing.py", WORK/"splicing_snv.csv",
    WORK/"splicing_dedup.csv", WORK/"dedup_audit.csv")
dedup = list(csv.DictReader((WORK/"splicing_dedup.csv").open()))
dkeys = [(r["chr"], r["pos"], r["ref"], r["alt"]) for r in dedup]
# both rows of the duplicated key are removed, not collapsed to one
assert len(dedup) == len(final) - 2 and len(dkeys) == len(set(dkeys)), dedup
print("OK strict de-duplication:", len(final), "->", len(dedup))
print("\nALL SMOKE TESTS PASSED")
