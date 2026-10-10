#!/usr/bin/env python3
"""Turn the per-scorer output of alphagenome_non_snv.py into the benchmark columns.

alphagenome_non_snv.py writes, for every RECOMMENDED_VARIANT_SCORERS entry, the count,
mean, min and max of its finite raw scores (and quantile scores where the API returns
them). This script derives the six global columns used in 03_datasets/:

  alphagenome_{raw,quantile}_score_max   max over the pool of the per-scorer max
  alphagenome_{raw,quantile}_score_min   min over the pool of the per-scorer min
  alphagenome_{raw,quantile}_score_mean  count-weighted mean over the pool of the
                                         per-scorer means

The pool is every recommended scorer except the '*_active' ones and contact_maps
(11 scorers). This rule reproduces the six global columns of the original pipeline
exactly (see 08_external_validation/scripts/03_assemble_sge_features.py).

alphagenome_splicing is not derived here: it comes from alphagenome_splicing.py, which
scores a differently centred interval. Rows whose API call failed (status != "ok") get
NaN.

Usage:
  python summarize_alphagenome.py --input-csv alphagenome.csv --output-csv out.csv
"""
import argparse

import numpy as np
import pandas as pd


def pool_of(frame):
    scorers = sorted({c[len("alphagenome_"):-len("_raw_n")]
                      for c in frame.columns
                      if c.startswith("alphagenome_") and c.endswith("_raw_n")})
    pool = [s for s in scorers if not s.endswith("_active") and s != "contact_maps"]
    if len(pool) != 11:
        raise RuntimeError(f"expected an 11-scorer pool, got {len(pool)}: {pool}")
    return pool


def summarize(frame):
    pool = pool_of(frame)
    out = frame[["chr", "pos", "ref", "alt"]].copy()
    ok = frame["status"].eq("ok").to_numpy() if "status" in frame else np.ones(len(frame), bool)
    for kind in ("raw", "quantile"):
        n = frame[[f"alphagenome_{s}_{kind}_n" for s in pool]].to_numpy(float)
        mean = frame[[f"alphagenome_{s}_{kind}_mean" for s in pool]].to_numpy(float)
        lo = frame[[f"alphagenome_{s}_{kind}_min" for s in pool]].to_numpy(float)
        hi = frame[[f"alphagenome_{s}_{kind}_max" for s in pool]].to_numpy(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"alphagenome_{kind}_score_max"] = np.where(ok, np.nanmax(hi, 1), np.nan)
            out[f"alphagenome_{kind}_score_min"] = np.where(ok, np.nanmin(lo, 1), np.nan)
            out[f"alphagenome_{kind}_score_mean"] = np.where(
                ok, np.nansum(n * mean, 1) / np.nansum(n, 1), np.nan)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", required=True, help="output of alphagenome_non_snv.py")
    ap.add_argument("--output-csv", required=True)
    a = ap.parse_args()
    frame = pd.read_csv(a.input_csv, low_memory=False)
    out = summarize(frame)
    out.to_csv(a.output_csv, index=False)
    print(f"{len(out)} variants, {int(out['alphagenome_raw_score_max'].notna().sum())} scored "
          f"-> {a.output_csv}")


if __name__ == "__main__":
    main()
