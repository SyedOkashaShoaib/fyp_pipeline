#!/usr/bin/env python3
"""
Rank candidate layers by how well a linear probe on their activations predicts
motif content (each selection motif's best-match score per window).

For each layer and summary (mean / max): ridge regression from the 1,500
standardised activation dimensions to each motif's standardised score,
evaluated by held-out R^2 under 5-fold cross-validation. Folds are grouped by
homoeologue/paralogue group so related promoters never sit on both sides of a
fold, and are identical for every layer. The regularisation strength is chosen
inside each training fold.

Fold score = held-out R^2 per motif, averaged within each motif similarity
group (so one large family cannot decide the layer), then across groups.

Decision rule (frozen in docs/layer_selection_protocol.md, 19 Oct 2026):
  best = highest mean score under the max summary among layers 1..n;
  a layer is tied if its mean per-fold difference from the best is within one
  SE of that paired difference; choose the tied layer nearest mid-stack, the
  lower one if two are equally near. Layer 0 is a reference only.

CPU only. Usage:
  python probe_layers.py --summaries data/probe/layer_summaries.npz \
      --targets data/probe/motif_best_scores.tsv \
      --motif-groups data/motifs/official_partition_candidate/motif_partition.tsv \
      --groups data/probe_sample_L1002.tsv --out data/probe/layer_probe_results.tsv
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

ALPHAS = np.logspace(-1, 5, 13)
N_FOLDS = 5


def r2(y, yhat):
    ss = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
    ss[ss == 0] = np.nan
    return 1 - ((y - yhat) ** 2).sum(axis=0) / ss


def decide(scores, n_layers):
    """scores: DataFrame layer x fold (max summary). Returns the decision."""
    mean = scores.mean(axis=1)
    cand = [k for k in scores.index if k != 0]
    best = max(cand, key=lambda k: mean[k])
    diff = scores.loc[best] - scores                     # per-fold paired difference
    dmean = diff.mean(axis=1)
    dse = diff.std(axis=1, ddof=1) / np.sqrt(scores.shape[1])
    tied = [k for k in scores.index if k == best or dmean[k] <= dse[k]]
    mid = n_layers / 2
    choice = min([k for k in tied if k != 0], key=lambda k: (abs(k - mid), k))
    hedge = [k for k in cand if k not in tied and dmean[k] <= 2 * dse[k]]
    return best, tied, choice, hedge, dmean, dse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summaries", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--motif-groups", required=True,
                    help="motif_partition.tsv with motif_id and group columns")
    ap.add_argument("--groups", required=True,
                    help="TSV with gene_id and group_id (the probe sample)")
    ap.add_argument("--n-layers", type=int, default=40)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    z = np.load(a.summaries, allow_pickle=False)
    ids = z["ids"].astype(str)
    layers = [int(k) for k in z["layers"]]
    tgt = pd.read_csv(a.targets, sep="\t", index_col=0).reindex(ids)
    if tgt.isna().any().any():
        raise SystemExit("targets missing for some windows; ids do not match")
    grp = pd.read_csv(a.groups, sep="\t").set_index("gene_id")["group_id"].reindex(ids)
    grp = grp.fillna(pd.Series(ids, index=ids))
    mg = pd.read_csv(a.motif_groups, sep="\t").set_index("motif_id")["group"]
    mgroup = mg.reindex(tgt.columns)
    if mgroup.isna().any():
        raise SystemExit(f"motifs without a similarity group: "
                         f"{list(tgt.columns[mgroup.isna()])[:5]}")
    mgroup = mgroup.to_numpy()

    Y = tgt.to_numpy(dtype=np.float64)
    folds = list(GroupKFold(n_splits=N_FOLDS).split(Y, groups=grp.to_numpy()))
    print(f"{len(ids)} windows, {Y.shape[1]} motif targets in "
          f"{len(set(mgroup))} similarity groups, layers {layers[0]}..{layers[-1]}, "
          f"{N_FOLDS} grouped folds")

    rows, motif_rows = [], []
    for summ in ("max", "mean"):
        for k in layers:
            X = z[f"{summ}_L{k}"].astype(np.float64)
            fold_scores, per_motif = [], []
            for tr, te in folds:
                mu, sd = X[tr].mean(0), X[tr].std(0)
                sd[sd == 0] = 1
                ymu, ysd = Y[tr].mean(0), Y[tr].std(0)
                ysd[ysd == 0] = 1
                m = RidgeCV(alphas=ALPHAS).fit((X[tr] - mu) / sd, (Y[tr] - ymu) / ysd)
                pred = m.predict((X[te] - mu) / sd)
                rm = r2((Y[te] - ymu) / ysd, pred)
                per_motif.append(rm)
                fold_scores.append(pd.Series(rm).groupby(mgroup).mean().mean())
            fs = np.array(fold_scores)
            rows.append({"summary": summ, "layer": k, "score": fs.mean(),
                         "se": fs.std(ddof=1) / np.sqrt(len(fs)),
                         "motif_mean_r2": np.nanmean(per_motif),
                         **{f"fold{i}": v for i, v in enumerate(fs)}})
            pm = np.nanmean(per_motif, axis=0)
            motif_rows += [{"summary": summ, "layer": k, "motif": c, "group": g,
                            "r2": v} for c, g, v in zip(tgt.columns, mgroup, pm)]
            print(f"  {summ:<4} layer {k:>2}: group-averaged R^2 {fs.mean():.4f} "
                  f"± {rows[-1]['se']:.4f}")

    res = pd.DataFrame(rows)
    fcols = [f"fold{i}" for i in range(N_FOLDS)]
    for summ in ("max", "mean"):
        sc = res[res.summary == summ].set_index("layer")[fcols]
        best, tied, choice, hedge, dmean, dse = decide(sc, a.n_layers)
        sel = res.summary == summ
        res.loc[sel, "diff_from_best"] = res.loc[sel, "layer"].map(dmean).to_numpy()
        res.loc[sel, "diff_se"] = res.loc[sel, "layer"].map(dse).to_numpy()
        res.loc[sel, "tied"] = res.loc[sel, "layer"].isin(tied).to_numpy()
        tag = "" if summ == "max" else " (secondary, does not decide)"
        print(f"\n[{summ}]{tag} best layer {best}; tie set {sorted(tied)}; "
              f"choice under the rule: layer {choice}; hedge {hedge}")
        if 0 in tied:
            print(f"NOTE [{summ}]: layer 0 (embedding) is tied with the best; motif "
                  "content may not need context. Report this prominently.")

    res.to_csv(a.out, sep="\t", index=False, lineterminator="\n")
    mout = a.out.replace(".tsv", "_per_motif.tsv")
    pd.DataFrame(motif_rows).to_csv(mout, sep="\t", index=False,
                                    float_format="%.5f", lineterminator="\n")
    print(f"wrote {a.out} and {mout}")


if __name__ == "__main__":
    main()
