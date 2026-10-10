"""
Build manuscript Tables 1-10 from the result files.

Each table is written as a CSV with full-precision values, and all ten are
rendered in TABLES.md with the rounding used in the manuscript, so a printed
number can be traced to the file it comes from.

  Table 1   leading individual features            05_results/{task}_singlefeature.csv
  Table 2   MERGE and AlphaGenome, fixed cohorts   05_results/{task}_merge_summary.csv,
                                                   05_results_alphagenome_average/
  Table 3   paired tests on pooled OOF             05_results/stats/paired_tests.csv
  Table 4   class-balance sensitivity              05_results/balanced_1to1/
  Table 5   calibration                            05_results/calibration/calibration.json
  Table 6   operating point at 0.5                 05_results/{task}_merge_summary.csv
  Table 7   established predictors, common subset  07_baselines/results/baseline_comparison.csv
  Table 8   paired differences, cluster bootstrap  07_baselines/results/significance.csv
  Table 9   BRCA1 saturation genome editing        08_external_validation/results/
  Table 10  dataset composition                    03_datasets/

Usage: python manuscript_tables.py [--out DIR]     (default 05_results/tables)
"""
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import AGAVG, BALANCED, DATA, ENSEMBLE_EXCLUDE_PREFIXES, ID_COLS, REPO, RESULTS, TASKS

R = Path(RESULTS)
TASK_LABEL = {"coding": "Coding SNVs", "splicing": "Splicing SNVs",
              "noncoding": "Noncoding SNVs", "nonsnv": "Non-SNVs"}
STRATEGY = {"beam": "beam search", "lasso": "LASSO", "correlation": "correlation filtering"}
NAME = {
    "AlphaMissense_score": "AlphaMissense", "ESM1b_score": "ESM-1b", "GPN_MSA_score": "GPN-MSA",
    "carbon_score": "Carbon-3B", "evo2_7b_score": "Evo 2 7B", "evo2_7b_base_score": "Evo 2 7B-base",
    "evo2_1b_base_score": "Evo 2 1B-base", "alphagenome_splicing": "AlphaGenome splicing",
    "alphagenome_raw_score_max": "AlphaGenome raw maximum",
    "alphagenome_quantile_score_max": "AlphaGenome quantile maximum",
    "alphagenome_raw_score_mean": "AlphaGenome raw mean",
    "AlphaGenome_raw_max": "AlphaGenome raw maximum",
    "AlphaGenome_quantile_max": "AlphaGenome quantile maximum",
    "Polyphen2_HVAR": "PolyPhen-2 HVAR",
}
SGE = REPO / "08_external_validation" / "results"


def name(x):
    return NAME.get(x, x)


def meta(task):
    return json.loads((R / f"{task}_meta.json").read_text())


def best_row(directory, task):
    m = json.loads((Path(directory) / f"{task}_meta.json").read_text())
    s = pd.read_csv(Path(directory) / f"{task}_merge_summary.csv")
    return m, s[s.strategy == m["best_strategy"]].iloc[0]


def table1():
    rows = []
    for t in TASKS:
        sf = pd.read_csv(R / f"{t}_singlefeature.csv").sort_values("auroc_mean", ascending=False)
        for rank, r in enumerate(sf.head(3).itertuples(), 1):
            rows.append({"task": TASK_LABEL[t], "rank": rank, "feature": name(r.feature),
                         "column": r.feature, "auroc": r.auroc_mean, "auroc_sd": r.auroc_sd,
                         "auprc": r.auprc_mean, "auprc_sd": r.auprc_sd})
    return pd.DataFrame(rows)


def table2():
    avg = pd.read_csv(Path(AGAVG) / "alphagenome_average_summary.csv")
    rows = []
    for t in TASKS:
        m, r = best_row(R, t)
        rows.append({"task": TASK_LABEL[t], "score": f"MERGE ({STRATEGY[m['best_strategy']]})",
                     "auroc": r.auroc_mean, "auroc_sd": r.auroc_sd,
                     "auprc": r.auprc_mean, "auprc_sd": r.auprc_sd})
        if t == "splicing":
            ag = ["alphagenome_splicing"]
        elif t == "noncoding":
            ag = ["alphagenome_raw_score_max", "alphagenome_quantile_score_max"]
        else:
            continue
        sf = pd.read_csv(R / f"{t}_singlefeature.csv").set_index("feature")
        for f in ag:
            rows.append({"task": TASK_LABEL[t], "score": name(f),
                         "auroc": sf.loc[f, "auroc_mean"], "auroc_sd": sf.loc[f, "auroc_sd"],
                         "auprc": sf.loc[f, "auprc_mean"], "auprc_sd": sf.loc[f, "auprc_sd"]})
        a = avg[(avg.task == t) & (avg.predictor == "MERGE + AlphaGenome average")].iloc[0]
        rows.append({"task": TASK_LABEL[t], "score": "MERGE + AlphaGenome average",
                     "auroc": a.auroc, "auroc_sd": a.auroc_sd,
                     "auprc": a.auprc, "auprc_sd": a.auprc_sd})
    return pd.DataFrame(rows)


def table3():
    d = pd.read_csv(R / "stats" / "paired_tests.csv")
    d.insert(2, "reference_name", d["reference"].map(name))
    return d[["task", "comparison", "reference_name", "reference", "n",
              "delta_auroc", "delta_auroc_ci_lo", "delta_auroc_ci_hi", "delong_p",
              "delta_auprc", "delta_auprc_ci_lo", "delta_auprc_ci_hi"]]


def table4():
    rows = []
    for t in TASKS:
        mo, ro = best_row(R, t)
        mb, rb = best_row(BALANCED, t)
        rows.append({"task": TASK_LABEL[t], "variants_after_downsampling": mb["n"],
                     "auroc_original": ro.auroc_mean, "auroc_balanced": rb.auroc_mean,
                     "auprc_original": ro.auprc_mean, "auprc_balanced": rb.auprc_mean,
                     "no_skill_auprc_original": mo["pos_rate"],
                     "no_skill_auprc_balanced": mb["pos_rate"],
                     "strategy_original": mo["best_strategy"],
                     "strategy_balanced": mb["best_strategy"]})
    return pd.DataFrame(rows)


def table5():
    cal = {c["task"]: c for c in json.load(open(R / "calibration" / "calibration.json"))}
    return pd.DataFrame([{"task": TASK_LABEL[t], "positive_prevalence": meta(t)["pos_rate"],
                          "brier": cal[t]["brier"], "brier_base_rate": cal[t]["brier_baseline"],
                          "ece": cal[t]["ece"]} for t in TASKS])


def table6():
    rows = []
    for t in TASKS:
        _, r = best_row(R, t)
        rows.append({"task": TASK_LABEL[t], "balanced_accuracy": r.bacc, "f1": r.f1,
                     "precision": r.precision, "recall": r.recall,
                     "specificity": r.specificity, "mcc": r.mcc})
    return pd.DataFrame(rows)


TABLE7 = {
    "coding": ["MetaRNN", "ClinPred", "REVEL", "VARITY", "VEST4", "MERGE", "BayesDel", "CADD",
               "AlphaGenome"],
    "splicing": ["AlphaGenome", "MERGE + AlphaGenome average", "SpliceAI", "Pangolin", "MERGE",
                 "CADD"],
    "noncoding": ["MERGE + AlphaGenome average", "AlphaGenome_quantile_max",
                  "AlphaGenome_raw_max", "MERGE", "CADD", "ncER", "LINSIGHT"],
    "nonsnv": ["MERGE", "CADD", "Pangolin", "AlphaGenome"],
}


def table7():
    b = pd.read_csv(REPO / "07_baselines" / "results" / "baseline_comparison.csv")
    rows = []
    for t, preds in TABLE7.items():
        for p in preds:
            r = b[(b.task == t) & (b.predictor == p)].iloc[0]
            group = ("MERGE" if p == "MERGE" else "Fixed average" if p.startswith("MERGE")
                     else r.tier)
            rows.append({"task": TASK_LABEL[t], "predictor": name(p), "group": group,
                         "auroc": r.auroc_common, "auprc": r.auprc_common})
    return pd.DataFrame(rows)


def table8():
    s = pd.read_csv(REPO / "07_baselines" / "results" / "significance.csv")
    s.insert(0, "score", s.pop("anchor").replace({"MERGE + AlphaGenome average": "Average"}))
    s["task"] = s["task"].map(TASK_LABEL)
    s["comparator"] = s["comparator"].map(lambda x: name("AlphaGenome splicing")
                                          if x == "AlphaGenome" else name(x))
    return s


TABLE9 = ["MERGE without AlphaGenome", "AlphaMissense_score", "VEST4", "BayesDel", "VARITY",
          "REVEL", "ESM1b_score", "ClinPred", "PolyPhen-2 HVAR", "CADD",
          "alphagenome_raw_score_max"]


def table9():
    m = pd.read_csv(SGE / "sge_external_validation_metrics.csv").set_index("predictor")
    d = pd.read_csv(SGE / "sge_paired_bootstrap_deltas.csv")
    d["predictor"] = d["comparison"].str.replace("MERGE without AlphaGenome - ", "", regex=False)
    d = d.set_index("predictor")
    rows = []
    for p in TABLE9:
        r = {"predictor": "MERGE" if p.startswith("MERGE") else name(p), "n": m.loc[p, "n_common"],
             "auroc": m.loc[p, "auroc"], "auroc_lo": m.loc[p, "auroc_ci_low"],
             "auroc_hi": m.loc[p, "auroc_ci_high"], "auprc": m.loc[p, "auprc"],
             "auprc_lo": m.loc[p, "auprc_ci_low"], "auprc_hi": m.loc[p, "auprc_ci_high"]}
        if p in d.index:
            x = d.loc[p]
            r.update({"delta_auroc": x.delta_auroc, "delta_auroc_lo": x.delta_auroc_ci_low,
                      "delta_auroc_hi": x.delta_auroc_ci_high, "p_auroc": x.p_auroc,
                      "delta_auprc": x.delta_auprc, "delta_auprc_lo": x.delta_auprc_ci_low,
                      "delta_auprc_hi": x.delta_auprc_ci_high, "p_auprc": x.p_auprc})
        rows.append(r)
    return pd.DataFrame(rows)


def table10():
    rows = []
    for t, f in TASKS.items():
        df = pd.read_csv(Path(DATA) / f, low_memory=False)
        feats = [c for c in df.columns if c not in ID_COLS]
        pool = [c for c in feats if not any(c.startswith(p) for p in ENSEMBLE_EXCLUDE_PREFIXES)]
        rows.append({"task": TASK_LABEL[t], "variants": len(df),
                     "positive": int(df.label.sum()), "negative": int((df.label == 0).sum()),
                     "positive_prevalence": df.label.mean(),
                     "benchmark_features": len(feats), "merge_candidate_features": len(pool)})
    return pd.DataFrame(rows)


# ---------------- rendering ---------------------------------------------------
def f4(x):
    return f"{x:.4f}".replace("-", "−")


def f3(x):
    return f"{x:.3f}".replace("-", "−")


def ci(lo, hi, f=f4):
    return f"({f(lo)} to {f(hi)})"


def pval(p, bootstrap=False):
    if bootstrap:
        return "< 0.001" if p < 0.001 else f"{p:.3f}"
    if p < 1e-16:
        return "< 10^-16"
    mant, exp = f"{p:.1e}".split("e")
    return f"{mant} x 10^{int(exp)}"


def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(out)


def render(t):
    parts = ["# Manuscript tables\n",
             "Generated by `04_modeling/manuscript_tables.py`; full-precision values are in the "
             "CSV next to this file. Means and sample SDs are over the three chromosome-held-out "
             "outer folds unless stated otherwise.\n"]

    d = t[1]
    parts += ["## Table 1. Leading individual features by mean outer-fold AUROC\n", md(pd.DataFrame({
        "Task": d.task, "Rank": d["rank"], "Feature": d.feature,
        "AUROC / AUPRC": [f"{a:.4f} / {b:.4f}" for a, b in zip(d.auroc, d.auprc)]}))]

    d = t[2]
    parts += ["\n## Table 2. MERGE and independent AlphaGenome comparisons\n", md(pd.DataFrame({
        "Task": d.task, "Score": d.score,
        "AUROC mean ± SD": [f"{a:.4f} ± {s:.4f}" for a, s in zip(d.auroc, d.auroc_sd)],
        "AUPRC mean ± SD": [f"{a:.4f} ± {s:.4f}" for a, s in zip(d.auprc, d.auprc_sd)]}))]

    d = t[3]
    parts += ["\n## Table 3. Paired comparisons on pooled out-of-fold predictions\n", md(pd.DataFrame({
        "Task": d.task.map(TASK_LABEL), "Comparison": d.comparison, "Reference": d.reference_name,
        "ΔAUROC (95% CI)": [f"{f4(a)} {ci(l, h)}" for a, l, h in
                             zip(d.delta_auroc, d.delta_auroc_ci_lo, d.delta_auroc_ci_hi)],
        "DeLong P": d.delong_p.map(pval),
        "ΔAUPRC (95% CI)": [f"{f4(a)} {ci(l, h)}" for a, l, h in
                             zip(d.delta_auprc, d.delta_auprc_ci_lo, d.delta_auprc_ci_hi)]}))]

    d = t[4]
    parts += ["\n## Table 4. Class-balance sensitivity\n", md(pd.DataFrame({
        "Task": d.task, "Variants after downsampling": d.variants_after_downsampling,
        "AUROC (orig. → 1:1)": [f"{a:.4f} → {b:.4f}" for a, b in zip(d.auroc_original, d.auroc_balanced)],
        "AUPRC (orig. → 1:1)": [f"{a:.4f} → {b:.4f}" for a, b in zip(d.auprc_original, d.auprc_balanced)],
        "Orig. no-skill AUPRC": d.no_skill_auprc_original.map("{:.4f}".format),
        "1:1 no-skill AUPRC": d.no_skill_auprc_balanced.map("{:.4f}".format)}))]

    d = t[5]
    parts += ["\n## Table 5. Calibration on pooled out-of-fold predictions\n", md(pd.DataFrame({
        "Task": d.task, "Positive prevalence": d.positive_prevalence.map("{:.4f}".format),
        "Brier score": d.brier.map("{:.4f}".format),
        "Brier (base-rate predictor)": d.brier_base_rate.map("{:.4f}".format),
        "ECE": d.ece.map("{:.3f}".format)}))]

    d = t[6]
    parts += ["\n## Table 6. Operating performance at a score threshold of 0.5\n", md(pd.DataFrame({
        "Task": d.task, "Balanced acc.": d.balanced_accuracy.map(f3), "F1": d.f1.map(f3),
        "Precision": d.precision.map(f3), "Recall": d.recall.map(f3),
        "Specificity": d.specificity.map(f3), "MCC": d.mcc.map(f3)}))]

    d = t[7]
    parts += ["\n## Table 7. Established predictors on common scored subsets\n", md(pd.DataFrame({
        "Task": d.task, "Predictor": d.predictor, "Group": d.group,
        "AUROC": d.auroc.map("{:.4f}".format), "AUPRC": d.auprc.map("{:.4f}".format)}))]

    d = t[8]
    parts += ["\n## Table 8. Paired differences from established predictors "
              "(chromosome-cluster bootstrap)\n", md(pd.DataFrame({
                  "Task": d.task, "Score and comparator": d.score + " vs " + d.comparator, "n": d.n,
                  "ΔAUROC (95% CI)": [f"{f4(a)} {ci(l, h)}" for a, l, h in
                                       zip(d.d_auroc, d.auroc_lo, d.auroc_hi)],
                  "ΔAUPRC (95% CI)": [f"{f4(a)} {ci(l, h)}" for a, l, h in
                                       zip(d.d_auprc, d.auprc_lo, d.auprc_hi)]}))]

    d = t[9]
    delta = lambda a, l, h: "" if pd.isna(a) else f"{f3(a)} {ci(l, h, f3)}"
    parts += ["\n## Table 9. BRCA1 saturation genome editing (n = 1,421)\n", md(pd.DataFrame({
        "Predictor": d.predictor,
        "AUROC (95% CI)": [f"{f3(a)} {ci(l, h, f3)}" for a, l, h in zip(d.auroc, d.auroc_lo, d.auroc_hi)],
        "AUPRC (95% CI)": [f"{f3(a)} {ci(l, h, f3)}" for a, l, h in zip(d.auprc, d.auprc_lo, d.auprc_hi)],
        "ΔAUROC (95% CI)": [delta(a, l, h) for a, l, h in
                             zip(d.delta_auroc, d.delta_auroc_lo, d.delta_auroc_hi)],
        "P (ΔAUROC)": ["" if pd.isna(p) else pval(p, True) for p in d.p_auroc],
        "ΔAUPRC (95% CI)": [delta(a, l, h) for a, l, h in
                             zip(d.delta_auprc, d.delta_auprc_lo, d.delta_auprc_hi)],
        "P (ΔAUPRC)": ["" if pd.isna(p) else pval(p, True) for p in d.p_auprc]}))]

    d = t[10]
    parts += ["\n## Table 10. Composition of the four benchmark datasets\n", md(pd.DataFrame({
        "Task": d.task, "Variants": d.variants,
        "Positive / negative": [f"{p} / {n}" for p, n in zip(d.positive, d.negative)],
        "Positive prevalence": d.positive_prevalence.map("{:.2%}".format),
        "Benchmark features": d.benchmark_features,
        "MERGE candidate features": d.merge_candidate_features}))]
    return "\n".join(parts) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(R / "tables"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    builders = [table1, table2, table3, table4, table5, table6, table7, table8, table9, table10]
    stems = ["leading_features", "merge_and_alphagenome", "paired_tests", "class_balance",
             "calibration", "operating_point", "established_predictors",
             "established_predictor_differences", "brca1_sge", "dataset_composition"]
    tables = {}
    for i, (build, stem) in enumerate(zip(builders, stems), 1):
        tables[i] = build()
        tables[i].to_csv(out / f"Table{i}_{stem}.csv", index=False)
    (out / "TABLES.md").write_text(render(tables))
    print("wrote", out)


if __name__ == "__main__":
    main()
