"""Assemble the SGE (Findlay 2018 BRCA1) feature table in the coding panel format.

Every model column is produced by the same script, checkpoint and parameters as the
original coding experiment (02_feature_extraction/shared/params.sh).  Nothing here
uses the SGE labels.

AlphaGenome six-column aggregation:
  pool  = RECOMMENDED_VARIANT_SCORERS minus every '*_active' scorer minus 'contact_maps'
  max   = max over pool of the per-scorer max
  min   = min over pool of the per-scorer min
  mean  = n-weighted mean over pool of the per-scorer means
This rule was reverse-engineered from the pipeline's own retained outputs
(alphagenome_work shards vs alphagenome_features.complete.csv) and reproduces all six
columns exactly on 36,876 variants (max |diff| 4.4e-16, 0 rows off by >1e-9).

ESM-1b orientation: the original coding table stores the RAW ESM-1b LLR (lower = more
damaging; as-is AUROC 0.091 on the coding set).  extract_sge_dbnsfp.py sign-flips it,
so it is negated back here to match the training orientation.
"""
from __future__ import annotations
import os, sys
import numpy as np, pandas as pd

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(EXP, "model_outputs")          # per-model outputs, one file per model
PANEL = ["heyenadna_score","NT_score","nt_v2_score","ESM1b_score","AlphaMissense_score",
         "GPN_MSA_score","alphagenome_raw_score_max","alphagenome_quantile_score_max",
         "alphagenome_raw_score_min","alphagenome_quantile_score_min",
         "alphagenome_raw_score_mean","alphagenome_quantile_score_mean",
         "carbon_score","evo2_7b_score","evo2_7b_base_score","evo2_1b_base_score",
         "generator_score","generator_v2_score","enformer_absmean","enformer_mean",
         "enformer_std","enformer_max_abs","enformer_top5_absmean","enformer_pos_sum",
         "enformer_neg_sum"]
KEY = ["chr","pos","ref","alt"]

def norm(df):
    df = df.copy(); df["chr"] = df["chr"].astype(str).str.replace("chr","",regex=False)
    df["pos"] = df["pos"].astype(int)
    for c in ("ref","alt"): df[c] = df[c].astype(str).str.upper()
    return df

def take(path, src, dst, chrom_col="chr"):
    d = pd.read_csv(path, low_memory=False)          # .csv.gz is read transparently
    if chrom_col != "chr": d = d.rename(columns={chrom_col: "chr"})
    d = norm(d)[KEY + [src]].rename(columns={src: dst})
    d[dst] = pd.to_numeric(d[dst], errors="coerce")
    return d.drop_duplicates(subset=KEY)

def alphagenome_six(path):
    d = pd.read_csv(path, low_memory=False)
    d = norm(d)
    scorers = sorted({c.split("_raw_")[0].replace("alphagenome_","")
                      for c in d.columns if c.endswith("_raw_n")})
    pool = [s for s in scorers if not s.endswith("_active") and s != "contact_maps"]
    if len(pool) != 11:
        raise RuntimeError(f"expected an 11-scorer pool, got {len(pool)}: {pool}")
    out = d[KEY].copy()
    for kind in ("raw","quantile"):
        n  = d[[f"alphagenome_{s}_{kind}_n"    for s in pool]].to_numpy(float)
        mu = d[[f"alphagenome_{s}_{kind}_mean" for s in pool]].to_numpy(float)
        mn = d[[f"alphagenome_{s}_{kind}_min"  for s in pool]].to_numpy(float)
        mx = d[[f"alphagenome_{s}_{kind}_max"  for s in pool]].to_numpy(float)
        with np.errstate(invalid="ignore"):
            out[f"alphagenome_{kind}_score_max"]  = np.nanmax(mx, 1)
            out[f"alphagenome_{kind}_score_min"]  = np.nanmin(mn, 1)
            out[f"alphagenome_{kind}_score_mean"] = np.nansum(n*mu,1)/np.nansum(n,1)
    return out.drop_duplicates(subset=KEY), pool

def main():
    cohort = pd.read_csv(os.path.join(EXP, "cohort", "sge_merge_external_validation_hg38.csv"),
                         low_memory=False)
    base = norm(cohort.rename(columns={"chrom": "chr"})[["variant_id","chr","pos","ref","alt","label"]])
    parts = []
    parts.append(take(f"{RAW}/hyenadna.csv.gz","score","heyenadna_score"))
    parts.append(take(f"{RAW}/nt.csv.gz","nt_mutation_score","NT_score"))
    parts.append(take(f"{RAW}/nt_v2.csv.gz","nt_v2_score","nt_v2_score"))
    parts.append(take(f"{RAW}/generator.csv.gz","generator_score","generator_score"))
    parts.append(take(f"{RAW}/generator_v2.csv.gz","generator_v2_score","generator_v2_score"))
    parts.append(take(f"{RAW}/carbon.csv.gz","carbon_score","carbon_score",chrom_col="chrom"))
    for m in ("evo2_7b","evo2_7b_base","evo2_1b_base"):
        parts.append(take(f"{RAW}/{m}.csv.gz","evo2_score",f"{m}_score"))
    enf = norm(pd.read_csv(f"{RAW}/enformer.csv.gz",low_memory=False))
    parts.append(enf[KEY+[c for c in enf.columns if c.startswith("enformer_")]]
                 .drop_duplicates(subset=KEY))
    ag, pool = alphagenome_six(f"{RAW}/alphagenome.csv.gz")
    parts.append(ag)
    print(f"[alphagenome] pool ({len(pool)}): {pool}")

    # --- precomputed lookups -------------------------------------------------
    gpn = norm(pd.read_csv(f"{RAW}/sge_GPN_MSA_score.csv"))[KEY+["GPN_MSA_score"]]
    parts.append(gpn.drop_duplicates(subset=KEY))
    db = pd.read_csv(f"{RAW}/sge_dbnsfp_scores.csv", low_memory=False)
    db = norm(db)[KEY+["AlphaMissense","ESM-1b"]].rename(
        columns={"AlphaMissense":"AlphaMissense_score","ESM-1b":"ESM1b_score"})
    db["AlphaMissense_score"] = pd.to_numeric(db["AlphaMissense_score"],errors="coerce")
    # un-flip: extract_sge_dbnsfp.py negates ESM1b; the coding panel stores the raw LLR
    db["ESM1b_score"] = -pd.to_numeric(db["ESM1b_score"],errors="coerce")
    parts.append(db.drop_duplicates(subset=KEY))

    out = base
    for p in parts:
        out = out.merge(p, on=KEY, how="left")
    missing = [c for c in PANEL if c not in out.columns]
    if missing: raise RuntimeError(f"missing panel columns: {missing}")
    out = out[["variant_id","chr","pos","ref","alt","label"]+PANEL]

    print(f"\nrows={len(out)}  dup_keys={out.duplicated(subset=KEY).sum()}")
    print(f"{'column':32s} {'coverage':>9s} {'min':>12s} {'max':>12s} {'mean':>12s}")
    for c in PANEL:
        s = pd.to_numeric(out[c],errors="coerce")
        print(f"  {c:30s} {s.notna().mean():>9.4f} {s.min():>12.4g} {s.max():>12.4g} {s.mean():>12.4g}")
    os.makedirs(os.path.join(EXP, "features"), exist_ok=True)
    out.to_csv(os.path.join(EXP, "features", "sge_features.csv"), index=False)
    print(f"\nwrote {os.path.join(EXP, 'features', 'sge_features.csv')}")

if __name__ == "__main__":
    main()
