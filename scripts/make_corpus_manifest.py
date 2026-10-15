#!/usr/bin/env python3
"""
Build the promoter component of the SAE training corpus, and draw the
benchmark sample from it.

A gene is in the corpus promoter component if:
  * it is a high-confidence gene with derived covariates (covariates.tsv),
  * its group (gene_groups_id85.tsv) contains NO evaluation-set gene,
  * its promoter window has n_frac <= 0.1 and full length.

The evaluation set is the union of the shoot and root analysis universes:
numeric CPM in all three conditions, CPM > 1 in all three, n_frac <= 0.1.

Outputs:
  corpus_promoters.tsv   one row per corpus promoter (the 45,119 expected)
  benchmark_sample.tsv   500 rows drawn from the above with the project seed
  exclusion_manifest.tsv every gene excluded from representation learning,
                         with the reason

Usage:
  python make_corpus_manifest.py --sd6 SD6.xlsx --covariates covariates.tsv \
      --groups gene_groups_id85.tsv --outdir data/
"""
import argparse
import hashlib
import os

import numpy as np
import pandas as pd

SEED = 20260829
MAX_N_FRAC = 0.1
PROMOTER_LEN = 1000
BENCHMARK_N = 500

SD6_COLS = ["GeneID", "Chr", "start", "end", "_x1",
            "S_DEG", "S_GFOLD", "S_edgeR", "S_CPM_WW", "S_CPM_DS1", "S_CPM_DS2",
            "S_CV_WW", "S_CV_DS1", "S_CV_DS2", "_x2",
            "R_DEG", "R_GFOLD", "R_edgeR", "R_CPM_WW", "R_CPM_DS1", "R_CPM_DS2",
            "R_CV_WW", "R_CV_DS1", "R_CV_DS2", "Ann"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def evaluation_set(sd6_path, cov):
    d = pd.read_excel(sd6_path, header=None, skiprows=4)
    d.columns = SD6_COLS
    d = d.set_index("GeneID")
    nfrac = cov["n_frac"].reindex(d.index)
    gap = nfrac > MAX_N_FRAC
    univ = pd.Series(False, index=d.index)
    for t in ("S", "R"):
        cols = [f"{t}_CPM_WW", f"{t}_CPM_DS1", f"{t}_CPM_DS2"]
        x = d[cols].apply(pd.to_numeric, errors="coerce")
        univ |= x.notna().all(axis=1) & (x > 1).all(axis=1) & ~gap
    return set(d.index[univ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sd6", required=True)
    ap.add_argument("--covariates", required=True)
    ap.add_argument("--groups", required=True)
    ap.add_argument("--outdir", default="data")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    cov = pd.read_csv(a.covariates, sep="\t").set_index("gene_id")
    groups = pd.read_csv(a.groups, sep="\t").set_index("gene_id")["group_id"]

    ev = evaluation_set(a.sd6, cov)
    ev_groups = set(groups.reindex(list(ev)).dropna())
    print(f"evaluation set: {len(ev)} genes in {len(ev_groups)} groups")

    g = cov.copy()
    g["group_id"] = groups.reindex(g.index)
    g.loc[g["group_id"].isna(), "group_id"] = g.index[g["group_id"].isna()]

    reason = pd.Series("", index=g.index, dtype=object)
    reason[g.index.isin(list(ev))] = "evaluation_gene"
    m = (reason == "") & g["group_id"].isin(ev_groups)
    reason[m] = "group_of_evaluation_gene"
    m = (reason == "") & ~(g["n_frac"] <= MAX_N_FRAC)
    reason[m] = "assembly_gap"
    m = (reason == "") & (g["prom_len"] != PROMOTER_LEN)
    reason[m] = "truncated_window"

    corpus = g[reason == ""].copy()
    excl = pd.DataFrame({"gene_id": reason.index[reason != ""],
                         "reason": reason[reason != ""].to_numpy()})

    print("excluded from representation learning (high-confidence genes):")
    print(excl["reason"].value_counts().to_string())
    print(f"corpus promoter component: {len(corpus)}")

    cols = ["chrom", "strand", "tss", "prom_start", "prom_end", "gc",
            "te_frac", "group_id"]
    corpus = corpus[cols].reset_index().sort_values(["chrom", "prom_start"])
    cpath = os.path.join(a.outdir, "corpus_promoters.tsv")
    corpus.to_csv(cpath, sep="\t", index=False)

    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(corpus), size=BENCHMARK_N, replace=False)
    bench = corpus.iloc[np.sort(pick)]
    bpath = os.path.join(a.outdir, "benchmark_sample.tsv")
    bench.to_csv(bpath, sep="\t", index=False)

    epath = os.path.join(a.outdir, "exclusion_manifest.tsv")
    excl.to_csv(epath, sep="\t", index=False)

    print("\nbenchmark sample by subgenome:",
          bench["chrom"].str[-1].value_counts().sort_index().to_dict())
    for p in (cpath, bpath, epath):
        print(f"{p}  sha256={sha256(p)[:16]}…")


if __name__ == "__main__":
    main()
