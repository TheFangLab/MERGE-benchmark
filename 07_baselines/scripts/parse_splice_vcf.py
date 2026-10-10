"""Reduce SpliceAI or Pangolin VCF output to a single score per variant.

SpliceAI's INFO field is
    SpliceAI=ALLELE|SYMBOL|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL
and the conventional summary is the largest of the four delta scores.

Pangolin's is
    Pangolin=GENE|pos:largest_increase|pos:largest_decrease|Warnings
and the conventional summary is the largest absolute change.

When several genes overlap a variant, the most damaging gene is kept, which is
the same convention both tools' own papers use for variant prioritisation.

Pangolin is run in CSV mode rather than VCF mode. Its VCF writer calls
`vcf.parser._Info` with the original PyVCF signature, and PyVCF3 -- the only
fork that installs on Python 3.11 -- added a `type_code` field to that
namedtuple, so the VCF path dies with "Info.__new__() missing 1 required
positional argument". The CSV path produces the same scores and avoids
patching an installed package.

Usage: python parse_splice_vcf.py <spliceai|pangolin> <in.vcf|in.csv> <out.tsv>
"""
import re
import sys


def spliceai_score(info):
    best = None
    for entry in re.findall(r"SpliceAI=([^;\t]+)", info):
        for record in entry.split(","):
            f = record.split("|")
            if len(f) < 6:
                continue
            vals = []
            for x in f[2:6]:
                try:
                    vals.append(float(x))
                except ValueError:
                    pass
            if vals:
                v = max(vals)
                best = v if best is None else max(best, v)
    return best


def pangolin_score(info):
    """Largest absolute score change across the entries.

    In VCF output the field arrives as "Pangolin=GENE|pos:up|pos:down|Warnings:";
    in CSV output the cell holds the same value with no "Pangolin=" prefix. Fall
    back to treating the whole string as one entry so both paths work.
    """
    best = None
    entries = re.findall(r"Pangolin=([^;\t]+)", info) or ([info] if info else [])
    for entry in entries:
        for record in entry.split(","):
            for part in record.split("|")[1:]:
                if ":" not in part:
                    continue
                try:
                    v = abs(float(part.split(":")[1]))
                except ValueError:
                    continue
                best = v if best is None else max(best, v)
    return best


def parse_csv(src, dst, score_of, tool):
    """Pangolin CSV output: the input columns plus a trailing Pangolin column."""
    import csv
    n_in = n_out = 0
    with open(src, newline="") as fh, open(dst, "w") as out:
        reader = csv.reader(fh)
        header = next(reader)
        col = {name: i for i, name in enumerate(header)}
        i_score = col.get("Pangolin", len(header) - 1)
        out.write("chr\tpos\tref\talt\tscore\n")
        for f in reader:
            n_in += 1
            if len(f) <= i_score:
                continue
            v = score_of(f[i_score])
            if v is None:
                continue
            chrom = f[col["CHROM"]].replace("chr", "")
            out.write(f"{chrom}\t{f[col['POS']]}\t{f[col['REF']]}"
                      f"\t{f[col['ALT']]}\t{v}\n")
            n_out += 1
    print(f"{tool}: {n_out}/{n_in} variants scored -> {dst}")


def parse_vcf(src, dst, score_of, tool):
    n_in = n_out = 0
    with open(src) as fh, open(dst, "w") as out:
        out.write("chr\tpos\tref\talt\tscore\n")
        for line in fh:
            if line.startswith("#"):
                continue
            n_in += 1
            f = line.rstrip("\n").split("\t")
            if len(f) < 8:
                continue
            v = score_of(f[7])
            if v is None:
                continue
            chrom = f[0].replace("chr", "")
            out.write(f"{chrom}\t{f[1]}\t{f[3]}\t{f[4]}\t{v}\n")
            n_out += 1
    print(f"{tool}: {n_out}/{n_in} variants scored -> {dst}")


def main():
    tool, src, dst = sys.argv[1], sys.argv[2], sys.argv[3]
    score_of = spliceai_score if tool == "spliceai" else pangolin_score
    if src.endswith(".csv"):
        parse_csv(src, dst, score_of, tool)
    else:
        parse_vcf(src, dst, score_of, tool)


if __name__ == "__main__":
    main()
