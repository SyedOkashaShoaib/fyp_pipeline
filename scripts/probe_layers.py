#!/usr/bin/env python3
"""
Rank candidate layers by how well a linear probe on their activations predicts
motif content (each selection motif's best-match score per window).

For each layer and summary (mean / max): ridge regression from the 1,500
standardised activation dimensions to each motif's standardised score,
evaluated by held-out R^2 under 5-fold cross-validation. Folds are grouped by
homoeologue/paralogue group so related promoters never sit on both sides of a
fold. The regularisation strength is chosen inside each training fold.

Layer score = mean held-out R^2 across motifs; its spread across folds gives a
standard error.

Decision rule (fix this BEFORE running; proposed default):
  choose the layer with the highest mean R^2 under the max summary;
  if another layer is within one standard error, choose whichever of the tied
  layers is nearest the middle of the stack.

CPU only. Usage:
  python probe_layers.py --summaries data/probe/layer_summaries.npz \
      --targets data/probe/motif_best_scores.tsv \
      --groups data/probe_sample.tsv --out data/probe/layer_probe_results.tsv
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summaries", required=True)
    ap.add_argument("--targets", required=True)
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

    Y = tgt.to_numpy(dtype=np.float64)
    folds = list(GroupKFold(n_splits=N_FOLDS).split(Y, groups=grp.to_numpy()))
    print(f"{len(ids)} windows, {Y.shape[1]} motif targets, layers {layers}, "
          f"{N_FOLDS} grouped folds")

    rows = []
    for summ in ("max", "mean"):
        for k in layers:
            X = z[f"{summ}_L{k}"].astype(np.float64)
            fold_scores = []
            for tr, te in folds:
                mu, sd = X[tr].mean(0), X[tr].std(0)
                sd[sd == 0] = 1
                ymu, ysd = Y[tr].mean(0), Y[tr].std(0)
                ysd[ysd == 0] = 1
                m = RidgeCV(alphas=ALPHAS).fit((X[tr] - mu) / sd, (Y[tr] - ymu) / ysd)
                pred = m.predict((X[te] - mu) / sd)
                fold_scores.append(np.nanmean(r2((Y[te] - ymu) / ysd, pred)))
            fs = np.array(fold_scores)
            rows.append({"summary": summ, "layer": k, "mean_r2": fs.mean(),
                         "se": fs.std(ddof=1) / np.sqrt(len(fs)),
                         "folds": " ".join(f"{v:.4f}" for v in fs)})
            print(f"  {summ:<4} layer {k:>2}: held-out R^2 {fs.mean():.4f} "
                  f"± {rows[-1]['se']:.4f}")

    res = pd.DataFrame(rows)
    res.to_csv(a.out, sep="\t", index=False)

    mx = res[res.summary == "max"].set_index("layer")
    top = mx["mean_r2"].idxmax()
    tied = mx.index[mx["mean_r2"] >= mx.loc[top, "mean_r2"] - mx.loc[top, "se"]]
    tied = [k for k in tied if k != 0]          # layer 0 is a reference only
    mid = a.n_layers / 2
    choice = min(tied, key=lambda k: (abs(k - mid), k)) if tied else top
    print(f"\nbest by max summary: layer {top}; within 1 SE: {sorted(tied)}")
    print(f"proposed choice under the stated rule: layer {choice}")
    if top == 0:
        print("NOTE: the embedding layer beats every transformer layer; the "
              "probe target may be too local to need context. Inspect before deciding.")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
