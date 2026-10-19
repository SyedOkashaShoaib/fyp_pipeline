#!/usr/bin/env python3
"""
Write matched case/control pairs for the drought-label layer test
(docs/layer_selection_protocol.md, secondary comparison).

Development split only, read from the frozen splits file (never redrawn).
Control pool N0, caliper 0.10, 1:1 without replacement, subgenome exact,
within-group pairs forbidden: the matching code of split_and_match.py,
with its own seed. Shoot is matched first, then root, from one generator.

Also writes the extraction manifest (union of paired genes, coordinates from
cov_dry.tsv) for extract_windows.py. Pairs whose windows contain non-ACGT
bases are dropped later, by filter_dev_pairs.py, once the windows exist.

Usage:
  python make_dev_pairs.py --sd6 data/SD6.xlsx --covariates data/cov_dry.tsv \
      --groups data/gene_groups_id85.tsv --splits "<splits.tsv>" --outdir data/drought
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from split_and_match import (BINARY, CALIPER, CONTINUOUS, build, load,  # noqa: E402
                             match, propensity, worst_smd)

PAIR_SEED = 20260831
SPLIT = "development"
POOL = "N0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sd6", required=True)
    ap.add_argument("--covariates", required=True)
    ap.add_argument("--groups", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--outdir", default="data/drought")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    d = load(a.sd6, a.covariates, a.groups)
    sp = pd.read_csv(a.splits, sep="\t").set_index("GeneID")
    ev = d.index[d["evaluation"]]
    if set(ev) != set(sp.index):
        raise SystemExit(f"evaluation set differs from the frozen split: "
                         f"{len(set(ev) ^ set(sp.index))} genes")
    for t in ("S", "R"):
        if not (d.loc[sp.index, f"univ_{t}"] == sp[f"univ_{t}"]).all():
            raise SystemExit(f"univ_{t} differs from the frozen split")
        if not (d.loc[sp.index, "group"].astype(str) == sp["group"].astype(str)).all():
            raise SystemExit("groups differ from the frozen split")
    d["split"] = sp["split"].reindex(d.index)
    print(f"frozen split reproduced: {len(sp)} evaluation genes, "
          f"{int((d['split'] == SPLIT).sum())} in {SPLIT}")

    rng = np.random.default_rng(PAIR_SEED)
    out = []
    for tissue, tname in [("S", "shoot"), ("R", "root")]:
        df = build(d, tissue, POOL, SPLIT)
        ps = propensity(df)
        ti, ci = match(df, ps, CALIPER, rng)
        nc = int(df["case"].sum())
        print(f"{tname}: cases {nc}, controls {len(df) - nc}, pairs {len(ti)}, "
              f"worst |SMD| {worst_smd(df, ti, ci):.3f}")
        p = pd.DataFrame({"tissue": tname,
                          "case_gene": df["GeneID"].to_numpy()[ti],
                          "control_gene": df["GeneID"].to_numpy()[ci],
                          "case_group": df["group"].to_numpy()[ti],
                          "control_group": df["group"].to_numpy()[ci],
                          "subgenome": df["subgenome"].to_numpy()[ti],
                          "ps_case": ps[ti], "ps_control": ps[ci]})
        for c in CONTINUOUS + BINARY:
            p[f"case_{c}"] = df[c].to_numpy()[ti]
            p[f"control_{c}"] = df[c].to_numpy()[ci]
        out.append(p)
    pairs = pd.concat(out, ignore_index=True)
    pairs.insert(0, "pair_id", [f"{t}_{i:05d}" for t, i in
                                zip(pairs["tissue"], pairs.groupby("tissue").cumcount())])
    pf = os.path.join(a.outdir, "dev_pairs_N0_seed20260831.tsv")
    pairs.to_csv(pf, sep="\t", index=False, lineterminator="\n")

    genes = pd.unique(pairs[["case_gene", "control_gene"]].to_numpy().ravel())
    cv = pd.read_csv(a.covariates, sep="\t").set_index("gene_id")
    man = cv.loc[genes, ["chrom", "strand", "tss", "prom_start", "prom_end"]]
    man.index.name = "gene_id"
    mf = os.path.join(a.outdir, "dev_window_manifest.tsv")
    man.reset_index().to_csv(mf, sep="\t", index=False, lineterminator="\n")
    print(f"wrote {len(pairs)} pairs to {pf}\nwrote {len(man)} genes to {mf}")


if __name__ == "__main__":
    main()
