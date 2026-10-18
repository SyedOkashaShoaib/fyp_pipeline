#!/usr/bin/env python3
"""
Compute layer-selection probe targets: for each probe window and each
selection-set motif, the best match score anywhere in the window, either strand.

Why a continuous score rather than present/absent: any fixed cutoff trades
false positives against misses, and the right cutoff differs per motif. On a
1 kb window (~2,000 scored positions across both strands) a per-position
false-positive rate of 1e-4 labels ~18% of windows positive for every motif by
chance, while 1e-5 misses most genuine sites of short motifs. The best-match
score needs no cutoff: windows containing a real site score higher on average,
and a layer that encodes motif content predicts the score better.

Scores are log-odds against the base composition of the probe windows
themselves. A present/absent table at --fpr is also written, for inspection
only; the probe uses the continuous scores.

CPU only. Usage:
  python scan_motifs.py --windows data/probe_windows.fa \
      --meme data/motifs/motifs_selection.meme --outdir data/probe
"""
import argparse
import os

import numpy as np
import pandas as pd
from Bio import motifs
from Bio.Seq import Seq

PSEUDOCOUNT = 0.5
DIAG_FPR = 1e-4


def read_fasta(path):
    ids, seqs, cur, name = [], [], [], None
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            if name is not None:
                ids.append(name)
                seqs.append("".join(cur))
            name, cur = line[1:].split()[0], []
        elif line:
            cur.append(line.upper())
    if name is not None:
        ids.append(name)
        seqs.append("".join(cur))
    return ids, seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True)
    ap.add_argument("--meme", required=True)
    ap.add_argument("--outdir", default="data/probe")
    ap.add_argument("--fpr", type=float, default=DIAG_FPR,
                    help="cutoff for the inspection-only present/absent table")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    ids, seqs = read_fasta(a.windows)
    if any("N" in s for s in seqs):
        raise SystemExit("probe windows contain N; regenerate the probe sample")
    joined = "".join(seqs)
    bg = {b: joined.count(b) / len(joined) for b in "ACGT"}
    print(f"{len(seqs)} windows; background "
          + " ".join(f"{b}={bg[b]:.3f}" for b in "ACGT"))

    mots = list(motifs.parse(open(a.meme), "minimal"))
    tf_name = {}
    for line in open(a.meme):          # Biopython keeps only the matrix ID
        if line.startswith("MOTIF"):
            parts = line.split()
            tf_name[parts[1]] = parts[2] if len(parts) > 2 else parts[1]
    print(f"{len(mots)} selection motifs")

    seq_objs = [Seq(s) for s in seqs]
    best = np.zeros((len(seqs), len(mots)), dtype=np.float32)
    hit = np.zeros((len(seqs), len(mots)), dtype=np.uint8)
    info = []
    for j, m in enumerate(mots):
        pssm = m.counts.normalize(pseudocounts=PSEUDOCOUNT).log_odds(bg)
        rc = pssm.reverse_complement()
        thr = pssm.distribution(background=bg, precision=10 ** 3).threshold_fpr(a.fpr)
        for i, s in enumerate(seq_objs):
            v = max(float(np.max(pssm.calculate(s))), float(np.max(rc.calculate(s))))
            best[i, j] = v
            hit[i, j] = v >= thr
        info.append({"motif": m.name, "tf_name": tf_name.get(m.name, m.name),
                     "length": m.length, "max_possible": float(pssm.max),
                     "best_mean": float(best[:, j].mean()),
                     "best_sd": float(best[:, j].std()),
                     "present_at_fpr": float(hit[:, j].mean())})
        if (j + 1) % 20 == 0:
            print(f"  scanned {j + 1}/{len(mots)} motifs")

    cols = [m.name for m in mots]
    out = pd.DataFrame(best, index=ids, columns=cols)
    out.index.name = "gene_id"
    out.to_csv(os.path.join(a.outdir, "motif_best_scores.tsv"), sep="\t",
               float_format="%.4f")
    h = pd.DataFrame(hit, index=ids, columns=cols)
    h.index.name = "gene_id"
    h.to_csv(os.path.join(a.outdir, f"motif_present_fpr{a.fpr:g}.tsv"), sep="\t")
    pd.DataFrame(info).to_csv(os.path.join(a.outdir, "motif_summary.tsv"),
                              sep="\t", index=False)

    sd = np.array([r["best_sd"] for r in info])
    print(f"\nbest-score spread across windows (sd): min {sd.min():.2f}  "
          f"median {np.median(sd):.2f}  max {sd.max():.2f}")
    print(f"present at fpr {a.fpr:g} (inspection only): median "
          f"{np.median([r['present_at_fpr'] for r in info]):.3f} of windows")
    print(f"wrote motif_best_scores.tsv (probe targets) and summaries to {a.outdir}/")


if __name__ == "__main__":
    main()
