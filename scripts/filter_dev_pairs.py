#!/usr/bin/env python3
"""
Drop every drought-test pair whose case or control window contains a non-ACGT
base, re-check covariate balance, and write the FASTA of windows still used.

Usage:
  python filter_dev_pairs.py --pairs data/drought/dev_pairs_N0_seed20260831.tsv \
      --windows data/drought/dev_windows_all.fa --outdir data/drought
"""
import argparse
import hashlib
import os

import numpy as np
import pandas as pd

COVS = ["log_cpm_ww", "cv_ww", "log_gene_len", "gc", "te_frac", "te_any",
        "prom_overlaps_gene"]


def read_fasta(path):
    recs, name = {}, None
    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith(">"):
            name = line[1:].split()[0]
            recs[name] = [line, []]
        elif line:
            recs[name][1].append(line)
    return {k: (h, "".join(s)) for k, (h, s) in recs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--outdir", default="data/drought")
    a = ap.parse_args()

    p = pd.read_csv(a.pairs, sep="\t")
    fa = read_fasta(a.windows)
    bad = {g for g, (_, s) in fa.items() if len(s) - sum(s.count(c) for c in "ACGT")}
    drop = p["case_gene"].isin(bad) | p["control_gene"].isin(bad)
    print(f"{len(bad)} windows contain non-ACGT bases")
    keep = p[~drop].reset_index(drop=True)
    for t, g in p.groupby("tissue"):
        k = keep[keep.tissue == t]
        smd = {c: (k[f"case_{c}"].mean() - k[f"control_{c}"].mean())
               / np.sqrt((k[f"case_{c}"].var() + k[f"control_{c}"].var()) / 2)
               for c in COVS}
        worst = max(smd, key=lambda c: abs(smd[c]))
        print(f"{t}: {len(g)} pairs, dropped {int(drop[g.index].sum())}, kept {len(k)}; "
              f"worst |SMD| {abs(smd[worst]):.4f} ({worst})")

    base = os.path.basename(a.pairs).replace(".tsv", "_noN.tsv")
    keep.to_csv(os.path.join(a.outdir, base), sep="\t", index=False, lineterminator="\n")
    genes = pd.unique(keep[["case_gene", "control_gene"]].to_numpy().ravel())
    h = hashlib.sha256()
    out = os.path.join(a.outdir, "dev_windows.fa")
    with open(out, "w", newline="\n") as f:
        for g in genes:
            hdr, s = fa[g]
            f.write(hdr + "\n")
            for i in range(0, len(s), 60):
                f.write(s[i:i + 60] + "\n")
            h.update(s.encode())
    print(f"wrote {len(keep)} pairs to {base} and {len(genes)} windows to {out}; "
          f"sequence-content sha256 {h.hexdigest()[:16]}")


if __name__ == "__main__":
    main()
