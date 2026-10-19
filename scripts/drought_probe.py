#!/usr/bin/env python3
"""
Drought-label layer test (secondary comparison; docs/layer_selection_protocol.md).

For each tissue, layer and summary: the case - control difference of pooled
activations goes into an L2 logistic regression with no intercept (the
conditional-logistic form). A seeded random half of the pairs has its sign
and label flipped so both classes are present. Columns are scaled on training
folds without centring; the penalty is chosen by grouped inner CV.

Folds: 5, grouped by connected components of the graph linking each case's
homologue group to its control's, so no group appears on both sides.

Metric: held-out paired accuracy, the share of pairs where the case scores
above its control (0.5 = chance), with a component-bootstrap 95% CI.
Baselines: layer 0 and a covariates-only model on the covariate differences.
Layers are compared by paired per-fold differences from the best layer.

This result cannot change the motif-chosen layer.

CPU only. Usage:
  python drought_probe.py --summaries data/drought/dev_layer_summaries.npz \
      --pairs data/drought/dev_pairs_N0_seed20260831_noN.tsv \
      --motif-layer K --out data/drought/drought_probe_results.tsv
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GroupKFold

SEED = 20260831
N_FOLDS = 5
INNER_FOLDS = 3
CS = np.logspace(-5, 1, 7)
N_BOOT = 2000
COVS = ["log_cpm_ww", "cv_ww", "log_gene_len", "gc", "te_frac", "te_any",
        "prom_overlaps_gene"]


def components(a, b):
    """Connected components of the graph with an edge a[i] - b[i]."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in zip(a, b):
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
    return np.array([find(u) for u in a])


def heldout_scores(X, flip, comp, folds):
    """Out-of-fold w.x for every pair (x = unflipped case - control)."""
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    Xf, y = X * flip[:, None], (flip > 0).astype(int)
    s = np.zeros(len(X))
    for tr, te in folds:
        sd = Xf[tr].std(0)
        sd[sd == 0] = 1
        inner = list(GroupKFold(n_splits=INNER_FOLDS).split(tr, groups=comp[tr]))
        m = LogisticRegressionCV(Cs=CS, cv=inner, fit_intercept=False,
                                 scoring="neg_log_loss", max_iter=3000)
        m.fit(Xf[tr] / sd, y[tr])
        s[te] = (X[te] / sd) @ m.coef_.ravel()
    return s


def summarise(s, folds, comp, rng_boot):
    hit = (s > 0).astype(float)
    fold_acc = [hit[te].mean() for _, te in folds]
    uc, inv = np.unique(comp, return_inverse=True)
    per_c = np.bincount(inv, weights=hit), np.bincount(inv)
    boots = []
    for _ in range(N_BOOT):
        w = np.bincount(rng_boot.integers(0, len(uc), len(uc)), minlength=len(uc))
        boots.append((w * per_c[0]).sum() / (w * per_c[1]).sum())
    return hit.mean(), np.percentile(boots, [2.5, 97.5]), fold_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summaries", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--motif-layer", type=int, default=None,
                    help="layer chosen by the motif probe, for the comparison")
    ap.add_argument("--summary-types", default="max,mean")
    ap.add_argument("--layers", default=None, help="subset, for testing only")
    ap.add_argument("--n-layers", type=int, default=40)
    ap.add_argument("--jobs", type=int, default=-2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    z = np.load(a.summaries, allow_pickle=False)
    ids = z["ids"].astype(str)
    pos = {g: i for i, g in enumerate(ids)}
    layers = [int(k) for k in z["layers"]]
    if a.layers:
        layers = [int(k) for k in a.layers.split(",")]
    P = pd.read_csv(a.pairs, sep="\t")
    summaries = a.summary_types.split(",")

    rows = []
    for tissue in ("root", "shoot"):
        p = P[P.tissue == tissue].reset_index(drop=True)
        ci = p["case_gene"].map(pos).to_numpy()
        co = p["control_gene"].map(pos).to_numpy()
        if np.isnan(ci.astype(float)).any() or np.isnan(co.astype(float)).any():
            raise SystemExit("some paired genes have no activations")
        comp = components(p["case_group"].astype(str), p["control_group"].astype(str))
        csize = pd.Series(comp).value_counts()
        rng = np.random.default_rng(SEED)
        flip = np.where(rng.random(len(p)) < 0.5, -1.0, 1.0)
        folds = list(GroupKFold(n_splits=N_FOLDS).split(p, groups=comp))
        print(f"\n{tissue}: {len(p)} pairs, {len(csize)} components "
              f"(largest {csize.iloc[0]} pairs); fold sizes "
              f"{[len(te) for _, te in folds]}")

        jobs = [("covariates", -1, np.column_stack(
            [p[f"case_{c}"] - p[f"control_{c}"] for c in COVS]).astype(np.float64))]
        jobs += [(s, k, None) for s in summaries for k in layers]

        def design(summ, k, X):
            if X is None:
                A = z[f"{summ}_L{k}"]
                X = (A[ci] - A[co]).astype(np.float64)
            return X

        scores = Parallel(n_jobs=a.jobs)(
            delayed(heldout_scores)(design(s, k, X), flip, comp, folds)
            for s, k, X in jobs)
        rb = np.random.default_rng(SEED + 1)
        for (summ, k, _), s in zip(jobs, scores):
            acc, ci95, fa = summarise(s, folds, comp, rb)
            rows.append({"tissue": tissue, "summary": summ, "layer": k,
                         "paired_acc": acc, "ci_lo": ci95[0], "ci_hi": ci95[1],
                         **{f"fold{i}": v for i, v in enumerate(fa)}})

    res = pd.DataFrame(rows)
    fcols = [f"fold{i}" for i in range(N_FOLDS)]
    mid = a.n_layers / 2
    for (tissue, summ), g in res[res.summary != "covariates"].groupby(["tissue", "summary"]):
        sc = g.set_index("layer")[fcols]
        mean = sc.mean(axis=1)
        best = max([k for k in sc.index if k != 0], key=lambda k: mean[k])
        diff = sc.loc[best] - sc
        dm, dse = diff.mean(axis=1), diff.std(axis=1, ddof=1) / np.sqrt(N_FOLDS)
        tied = [k for k in sc.index if k == best or dm[k] <= dse[k]]
        res.loc[g.index, "diff_from_best"] = g["layer"].map(dm).to_numpy()
        res.loc[g.index, "diff_se"] = g["layer"].map(dse).to_numpy()
        res.loc[g.index, "tied"] = g["layer"].isin(tied).to_numpy()
        cov = res[(res.tissue == tissue) & (res.summary == "covariates")].iloc[0]
        choice = min([k for k in tied if k != 0], key=lambda k: (abs(k - mid), k))
        print(f"\n[{tissue} / {summ}] best layer {best}: paired acc {mean[best]:.4f} "
              f"(CI {g.set_index('layer').loc[best, 'ci_lo']:.3f}-"
              f"{g.set_index('layer').loc[best, 'ci_hi']:.3f}); layer 0 {mean[0]:.4f}; "
              f"covariates-only {cov['paired_acc']:.4f}")
        print(f"  tie set ({len(tied)} layers): {sorted(tied)}; rule choice {choice}")
        if a.motif_layer is not None:
            m = a.motif_layer
            print(f"  motif-chosen layer {m}: acc {mean[m]:.4f}, behind best by "
                  f"{dm[m]:.4f} ± {dse[m]:.4f} (paired SE); "
                  f"{'tied with best' if m in tied else 'not tied'}; "
                  f"{'> 2 SE behind' if dm[m] > 2 * dse[m] else 'within 2 SE'}")

    res.to_csv(a.out, sep="\t", index=False, float_format="%.5f", lineterminator="\n")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
