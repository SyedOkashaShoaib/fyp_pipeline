#!/usr/bin/env python3
"""
Draw the layer-selection probe sample from the corpus promoter component.

Label-blind: drawn from corpus_promoters.tsv, which already excludes every
evaluation gene and every member of an evaluation gene's group. The benchmark
sample is excluded so the two remain independent. Windows containing any N
are excluded because N bases break both 6-mer tokenisation and motif scanning.

Usage:
  python make_probe_sample.py --corpus data/corpus_promoters.tsv \
      --benchmark data/benchmark_sample.tsv --covariates cov_dry.tsv \
      --n 4000 --out data/probe_sample.tsv
"""
import argparse
import numpy as np
import pandas as pd

SEED = 20260829

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", required=True)
ap.add_argument("--benchmark", required=True)
ap.add_argument("--covariates", required=True)
ap.add_argument("--n", type=int, default=4000)
ap.add_argument("--out", required=True)
a = ap.parse_args()

c = pd.read_csv(a.corpus, sep="\t")
b = set(pd.read_csv(a.benchmark, sep="\t")["gene_id"])
nf = pd.read_csv(a.covariates, sep="\t").set_index("gene_id")["n_frac"]
pool = c[~c["gene_id"].isin(b) & (c["gene_id"].map(nf) == 0)]
rng = np.random.default_rng(SEED + 1)        # distinct stream from the benchmark draw
pick = pool.iloc[np.sort(rng.choice(len(pool), size=a.n, replace=False))]
pick.to_csv(a.out, sep="\t", index=False)
print(f"eligible {len(pool)} of {len(c)} corpus promoters; wrote {len(pick)} to {a.out}")
print("by chromosome class:", pick["chrom"].str[-1].replace({"n": "Un"}).value_counts().to_dict())
