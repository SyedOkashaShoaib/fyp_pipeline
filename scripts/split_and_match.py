#!/usr/bin/env python3
"""
Build the frozen discovery/development/confirmation split and test whether
matching still works inside each split.

The split unit is the homoeolog/paralog group, so no group is divided across
splits and no evaluation promoter shares a group with a promoter in another
split. Assignment is size-aware: groups are placed largest-first into
whichever split is furthest below its target share, which keeps realised
proportions close to target despite indivisible blocks.

One split is built over the union of the shoot and root universes and reused
for both tissues, all layers, all autoencoder seeds and all baselines.

Usage:
  python split_and_match.py SD6.xlsx --covariates cov_dry.tsv \
      --groups gene_groups.tsv --out splits.tsv
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

SEED = 20260829
TARGETS = {"discovery": 0.60, "development": 0.20, "confirmation": 0.20}
CALIPER = 0.10
MAX_CANDIDATES = 25
CONTINUOUS = ["log_cpm_ww", "cv_ww", "log_gene_len", "gc", "te_frac"]
BINARY = ["te_any", "prom_overlaps_gene"]

SD6_COLS = ["GeneID", "Chr", "start", "end", "_x1",
            "S_DEG", "S_GFOLD", "S_edgeR",
            "S_CPM_WW", "S_CPM_DS1", "S_CPM_DS2",
            "S_CV_WW", "S_CV_DS1", "S_CV_DS2", "_x2",
            "R_DEG", "R_GFOLD", "R_edgeR",
            "R_CPM_WW", "R_CPM_DS1", "R_CPM_DS2",
            "R_CV_WW", "R_CV_DS1", "R_CV_DS2", "Ann"]


def load(sd6, cov, groups, max_n_frac=0.1):
    d = pd.read_excel(sd6, header=None, skiprows=4)
    d.columns = SD6_COLS
    for c in SD6_COLS:
        if "CPM" in c or "CV" in c:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d["subgenome"] = d["Chr"].str[-1]
    d["chr_group"] = d["Chr"].str[3]
    d = d.set_index("GeneID")

    cv = pd.read_csv(cov, sep="\t").set_index("gene_id").reindex(d.index)
    d["gc"], d["te_frac"] = cv["gc"], cv["te_frac"]
    d["te_any"] = (cv["te_frac"] > 0).astype(float)
    d["prom_overlaps_gene"] = cv["prom_overlaps_gene"].astype(float)
    d["gene_len"], d["n_frac"] = cv["gene_len"], cv["n_frac"]

    g = pd.read_csv(groups, sep="\t").set_index("gene_id")
    d["group"] = g["group_id"].reindex(d.index)
    d.loc[d["group"].isna(), "group"] = d.index[d["group"].isna()]

    gapped = d["n_frac"] > max_n_frac
    for t in ("S", "R"):
        cols = [f"{t}_CPM_WW", f"{t}_CPM_DS1", f"{t}_CPM_DS2"]
        d[f"univ_{t}"] = (d[cols].notna().all(axis=1)
                          & (d[cols] > 1).all(axis=1) & ~gapped)
    d["evaluation"] = d["univ_S"] | d["univ_R"]
    return d


def assign_splits(d, rng):
    """Size-aware greedy assignment of groups to splits."""
    ev = d[d["evaluation"]]
    counts = ev.groupby("group").size().sort_values(ascending=False)
    total = counts.sum()
    goal = {k: v * total for k, v in TARGETS.items()}
    filled = {k: 0 for k in TARGETS}
    assignment = {}

    # ties broken randomly so the result is seed-dependent, not order-dependent
    order = counts.index.to_numpy()
    jitter = rng.permutation(len(order))
    pairs = sorted(zip(counts.to_numpy(), jitter, order),
                   key=lambda x: (-x[0], x[1]))

    for size, _, grp in pairs:
        deficit = {k: goal[k] - filled[k] for k in TARGETS}
        pick = max(deficit, key=deficit.get)
        assignment[grp] = pick
        filled[pick] += size

    d["split"] = d["group"].map(assignment)
    print(f"split over {len(counts)} groups, {total} evaluation genes")
    for k in TARGETS:
        share = filled[k] / total
        print(f"  {k:<13} {filled[k]:>6} genes  {100 * share:5.2f}% "
              f"(target {100 * TARGETS[k]:.0f}%)")
    return d


def build(d, tissue, pool, split=None):
    univ = d[f"univ_{tissue}"]
    flag = d[f"{tissue}_DEG"].eq("Yes")
    edger = d[f"{tissue}_edgeR"].eq("Yes")
    gfold = d[f"{tissue}_GFOLD"].eq("Yes")
    case = univ & flag
    ctrl = {"N0": univ & ~flag, "N1": univ & ~edger & ~gfold,
            "N2": univ & ~edger}[pool]
    keep = case | ctrl
    if split:
        keep = keep & d["split"].eq(split)
    out = d.loc[keep, ["subgenome", "chr_group", "group", "gene_len", "gc",
                       "te_frac", "te_any", "prom_overlaps_gene"]].copy()
    out["case"] = case[keep].astype(int).to_numpy()
    out["log_cpm_ww"] = np.log1p(d.loc[keep, f"{tissue}_CPM_WW"])
    out["cv_ww"] = d.loc[keep, f"{tissue}_CV_WW"]
    out["log_gene_len"] = np.log10(d.loc[keep, "gene_len"].clip(lower=1))
    return out.dropna(subset=CONTINUOUS + BINARY).reset_index()


def smd(a, b):
    s = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return 0.0 if s == 0 else (a.mean() - b.mean()) / s


def propensity(df):
    X = pd.get_dummies(df[CONTINUOUS + BINARY + ["subgenome", "chr_group"]],
                       columns=["subgenome", "chr_group"],
                       drop_first=True).astype(float)
    X = (X - X.mean()) / X.std(ddof=0).replace(0, 1)
    m = LogisticRegression(max_iter=3000, C=1e6).fit(X.to_numpy(),
                                                     df["case"].to_numpy())
    p = m.predict_proba(X.to_numpy())[:, 1].clip(1e-6, 1 - 1e-6)
    lg = np.log(p / (1 - p))
    return (lg - lg.mean()) / lg.std(ddof=0)


class Avail:
    def __init__(self, n):
        self.n, self.rt, self.lf = n, np.arange(n + 2), np.arange(n + 2) - 1

    def _r(self, i):
        while i < self.n and self.rt[i] != i:
            self.rt[i] = self.rt[self.rt[i]]
            i = self.rt[i]
        return i

    def _l(self, i):
        while i >= 0 and self.lf[i] != i:
            self.lf[i] = self.lf[self.lf[i]]
            i = self.lf[i]
        return i

    def take(self, i):
        self.rt[i], self.lf[i] = i + 1, i - 1

    def candidates(self, pos, v, target, cal, limit):
        r, l, seen = self._r(min(pos, self.n)), self._l(min(pos, self.n) - 1), 0
        while seen < limit:
            dr = abs(v[r] - target) if r < self.n else np.inf
            dl = abs(v[l] - target) if 0 <= l < self.n else np.inf
            if not np.isfinite(min(dr, dl)) or min(dr, dl) > cal:
                return
            if dr <= dl:
                yield r
                r = self._r(r + 1)
            else:
                yield l
                l = self._l(l - 1)
            seen += 1


def match(df, ps, cal, rng):
    ti_all, ci_all = [], []
    grp = df["group"].to_numpy()
    pos_all = np.arange(len(df))
    for sg in sorted(df["subgenome"].dropna().unique()):
        pos = pos_all[df["subgenome"].to_numpy() == sg]
        isc = df["case"].to_numpy()[pos] == 1
        t, c = pos[isc], pos[~isc]
        if len(t) == 0 or len(c) == 0:
            continue
        o = np.argsort(ps[c])
        cv, cs = ps[c][o], c[o]
        av = Avail(len(cv))
        for k in rng.permutation(len(t)):
            ti = t[k]
            ins = int(np.searchsorted(cv, ps[ti]))
            ch = -1
            for j in av.candidates(ins, cv, ps[ti], cal, MAX_CANDIDATES):
                if grp[cs[j]] == grp[ti]:
                    continue
                ch = j
                break
            if ch < 0:
                continue
            av.take(ch)
            ti_all.append(ti)
            ci_all.append(cs[ch])
    return np.array(ti_all), np.array(ci_all)


def worst_smd(df, ti, ci):
    return max(abs(smd(df[c].to_numpy()[ti], df[c].to_numpy()[ci]))
               for c in CONTINUOUS + BINARY)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sd6")
    ap.add_argument("--covariates", required=True)
    ap.add_argument("--groups", required=True)
    ap.add_argument("--out", default="splits.tsv")
    a = ap.parse_args()

    rng = np.random.default_rng(SEED)
    d = load(a.sd6, a.covariates, a.groups)
    d = assign_splits(d, rng)
    d.loc[d["evaluation"], ["group", "split", "univ_S", "univ_R"]] \
        .to_csv(a.out, sep="\t")
    print(f"wrote {a.out}\n")

    print("split-then-match feasibility (caliper %.2f, 1:1, no replacement)"
          % CALIPER)
    for tissue, tname in [("S", "shoot"), ("R", "root")]:
        for pool in ["N0", "N2"]:
            print(f"\n  {tname} / {pool}")
            for split in ["ALL", "discovery", "development", "confirmation"]:
                df = build(d, tissue, pool, None if split == "ALL" else split)
                nc = int(df["case"].sum())
                if nc < 50 or len(df) - nc < 50:
                    print(f"    {split:<13} too small")
                    continue
                ps = propensity(df)
                ti, ci = match(df, ps, CALIPER, rng)
                print(f"    {split:<13} cases={nc:>6} ctrls={len(df) - nc:>6} "
                      f"pairs={len(ti):>6} unmatched={100 * (1 - len(ti) / nc):5.1f}%"
                      f"  worst|SMD|={worst_smd(df, ti, ci):.3f}")


if __name__ == "__main__":
    main()
