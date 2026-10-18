#!/usr/bin/env python3
"""
Partition JASPAR motifs into three disjoint sets: selection (layer choice),
discovery and confirmation (evaluation baselines and census).

Near-identical motifs (e.g. many MYB or WRKY variants) must not straddle sets,
or a motif used to pick the layer would have a twin in the evaluation set.
So motifs are first grouped by PWM similarity, then whole groups are split.

Similarity: best mean column-wise Pearson correlation over any core of
MIN_OVERLAP aligned columns, over all ungapped alignments, in both
orientations. Pairs at or above THRESHOLD are joined; groups are the
connected components. Groups are assigned largest-first to whichever set is
furthest below its target share, ties broken by the recorded seed.

Input: JASPAR CORE plants non-redundant, MEME format.
Usage:
  python partition_motifs.py --meme JASPAR20XX_CORE_plants_non-redundant_pfms_meme.txt \
      --outdir data/motifs
"""
import argparse
import os
import re

import numpy as np

SEED = 20260829
THRESHOLD = 0.90   # provisional: confirm against the diagnostics this prints
MIN_OVERLAP = 6
TARGETS = {"selection": 1 / 3, "discovery": 1 / 3, "confirmation": 1 / 3}


def read_meme(path):
    """Return list of (motif_id, name, matrix[L,4] in A,C,G,T order, raw_text)."""
    text = open(path).read()
    alph = re.search(r"ALPHABET\s*=\s*(\S+)", text)
    if alph and alph.group(1).upper() != "ACGT":
        raise SystemExit(f"unexpected alphabet {alph.group(1)}")
    header = text.split("MOTIF", 1)[0]
    motifs = []
    for block in re.split(r"(?m)^MOTIF\s+", text)[1:]:
        lines = block.strip().splitlines()
        parts = lines[0].split()
        mid, name = parts[0], (parts[1] if len(parts) > 1 else parts[0])
        rows, reading = [], False
        for ln in lines[1:]:
            if ln.startswith("letter-probability matrix"):
                reading = True
                continue
            if reading:
                vals = ln.split()
                if len(vals) == 4:
                    try:
                        rows.append([float(v) for v in vals])
                        continue
                    except ValueError:
                        pass
                if rows:
                    break
        m = np.array(rows)
        if m.ndim != 2 or m.shape[1] != 4 or len(m) == 0:
            raise SystemExit(f"could not parse matrix for {mid}")
        motifs.append((mid, name, m / m.sum(axis=1, keepdims=True),
                       "MOTIF " + block.rstrip() + "\n"))
    return header, motifs


def revcomp(m):
    return m[::-1, ::-1]           # reverse positions, swap A<->T and C<->G


def _std_cols(m):
    """Centre and scale each column (position) so column dot products are Pearson r."""
    x = m - m.mean(axis=1, keepdims=True)
    n = np.sqrt((x * x).sum(axis=1, keepdims=True))
    n[n == 0] = 1.0
    return x / n


def best_corr(a, b, w=None):
    """Best mean column-wise Pearson r over any WINDOW-column core alignment,
    in both orientations. Matching on a shared core, not the whole matrix,
    keeps a twin with an extra or altered flank from scoring low."""
    w = w or MIN_OVERLAP
    sa = _std_cols(a)
    best = -1.0
    for bb in (b, revcomp(b)):
        sb = _std_cols(bb)
        C = sa @ sb.T                      # C[i, j] = r(column i of a, column j of b)
        la, lb = C.shape
        for k in range(-(lb - w), la - w + 1):
            d = np.diagonal(C, offset=-k)  # a[i] aligned with b[i - k]
            if len(d) < w:
                continue
            cs = np.concatenate(([0.0], np.cumsum(d)))
            m = float(((cs[w:] - cs[:-w]) / w).max())
            if m > best:
                best = m
    return best


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meme", required=True)
    ap.add_argument("--outdir", default="data/motifs")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    header, motifs = read_meme(a.meme)
    n = len(motifs)
    print(f"{n} motifs read from {a.meme}")

    uf = UF(n)
    best_partner = np.full(n, -1.0)
    for i in range(n):
        for j in range(i + 1, n):
            r = best_corr(motifs[i][2], motifs[j][2])
            best_partner[i] = max(best_partner[i], r)
            best_partner[j] = max(best_partner[j], r)
            if r >= a.threshold:
                uf.union(i, j)
        if (i + 1) % 100 == 0:
            print(f"  compared {i + 1}/{n}")

    groups = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)
    sizes = sorted((len(v) for v in groups.values()), reverse=True)
    print(f"\n{len(groups)} similarity groups at r >= {a.threshold}; "
          f"largest {sizes[:5]}")

    # Diagnostics for choosing the threshold, label-blind. Too low a threshold
    # chains distinct families into one giant group (the same failure as
    # unfiltered paralogues); too high leaves obvious relatives split.
    q = np.percentile(best_partner, [10, 25, 50, 75, 90])
    print("each motif's most similar other motif, r percentiles "
          "10/25/50/75/90: " + " ".join(f"{v:.3f}" for v in q))
    if sizes[0] > 0.10 * n:
        print(f"WARNING: largest group holds {sizes[0]} of {n} motifs; "
              "the threshold is probably chaining families together")
    print("ten largest groups (member names, for a sanity check that "
          "relatives such as MYB, WRKY, NAC, bZIP or AP2/ERF co-group):")
    for g in sorted(groups, key=lambda g: -len(groups[g]))[:10]:
        names = [motifs[i][1] for i in groups[g]]
        print(f"  {len(names):>3}: {', '.join(names[:8])}"
              f"{' ...' if len(names) > 8 else ''}")

    # Assignment: largest groups first, each to the set furthest below its
    # target. Ties between sets (always the case for the first group, often
    # later) are broken at random from the recorded seed. Without this the
    # largest group would always land in the first-listed set (selection).
    rng = np.random.default_rng(SEED)
    order = sorted(groups, key=lambda g: (-len(groups[g]), rng.random()))
    goal = {k: v * n for k, v in TARGETS.items()}
    filled = {k: 0 for k in TARGETS}
    set_groups = {k: [] for k in TARGETS}
    assign = {}
    for g in order:
        deficit = {k: goal[k] - filled[k] for k in TARGETS}
        top = max(deficit.values())
        tied = [k for k in TARGETS if deficit[k] >= top - 1e-9]
        pick = tied[int(rng.integers(len(tied)))]
        for i in groups[g]:
            assign[i] = pick
        filled[pick] += len(groups[g])
        set_groups[pick].append(len(groups[g]))

    print("\nper-set composition (a set dominated by one group would let one "
          "binding pattern decide the layer):")
    for k in TARGETS:
        sz = sorted(set_groups[k], reverse=True)
        share = sz[0] / filled[k] if filled[k] else 0
        print(f"  {k:<13} {filled[k]:>4} motifs in {len(sz):>4} groups; "
              f"largest group {sz[0] if sz else 0} ({100 * share:.0f}% of set)")
        if share > 0.25:
            print(f"  WARNING: one group is over a quarter of the {k} set")

    with open(os.path.join(a.outdir, "motif_partition.tsv"), "w") as f:
        f.write("motif_id\tname\tlength\tgroup\tpartition\n")
        for i, (mid, name, m, _) in enumerate(motifs):
            f.write(f"{mid}\t{name}\t{len(m)}\t{motifs[uf.find(i)][0]}\t{assign[i]}\n")
    for part in TARGETS:
        with open(os.path.join(a.outdir, f"motifs_{part}.meme"), "w") as f:
            f.write(header)
            for i, mot in enumerate(motifs):
                if assign[i] == part:
                    f.write(mot[3] + "\n")
        print(f"  {part:<13} {filled[part]:>4} motifs")
    print(f"wrote {a.outdir}/motif_partition.tsv and three .meme files")


if __name__ == "__main__":
    main()
