#!/usr/bin/env python3
"""
Build the non-promoter region classes of the SAE training corpus (proposal
sec. 6.2, 6.4, handoff Task 10) and write their windows.

Every window is 1,002 bp, N-free, and overlaps no other corpus window, so the
classes never share bases with each other or with the promoter component.

Classes, in the order windows are claimed (scarcest first):
  intron      centred in an intron >= 1,002 bp of a corpus gene's .1 transcript;
              one per intron; gene strand
  utr5_tss    starts at the TSS of a corpus gene whose .1 transcript has an
              annotated 5' UTR and runs downstream; one per gene; gene strand
  splice      centred on one exon-intron boundary of a corpus gene's .1
              transcript, boundary and side drawn at random; one per gene;
              gene strand
  intergenic  >= --gene-pad bp from any HC or LC gene on a placed chromosome;
              matched to the promoter component on subgenome, TE fraction and
              GC; reference + strand
  repeat      inside one annotated repeat_region whose dominant TE superfamily
              is >= 80% of it, >= --gene-pad bp from any gene, on a placed
              chromosome; equal allocation across superfamilies, leftovers
              redistributed; the element's strand

Leakage rules:
  - genic classes draw only from corpus genes: high-confidence genes that are
    not in exclusion_manifest.tsv
  - no window overlaps the body or the 1,002 bp flanks of an evaluation gene
    or a member of an evaluation gene's group
  - no window has the same sequence as another window or a corpus promoter
  - SD6, splits and drought labels are never read

Outputs in --outdir:
  corpus_regions_L1002.tsv  one row per window (extract_activations.py manifest)
  corpus_regions_L1002.fa   windows, in the extract_windows.py header format
  build_report.json         seeds, input hashes, per-class counts and checks

Usage:
  python scripts/build_corpus_regions.py \
      --hc-gff <IWGSC_v1.1_HC_20170706.gff3> --lc-gff <IWGSC_v1.1_LC_20170706.gff3> \
      --te-gff <iwgsc_refseqv1.0_TransposableElements_2017Mar13.gff3> \
      --fasta data/161010_Chinese_Spring_v1.0_pseudomolecules.fasta \
      --promoters data/corpus_promoters_L1002.tsv \
      --promoter-windows data/corpus_windows_L1002.fa \
      --exclusion data/exclusion_manifest.tsv --outdir data/corpus_regions
"""
import argparse
import bisect
import hashlib
import json
import os
import re

import numpy as np
import pandas as pd
from pyfaidx import Fasta

L = 1002
SEED = 20261104
GFF_COLS = ["chrom", "src", "type", "start", "end", "score", "strand", "phase", "info"]
TE_BINS = [0, .001, .1, .25, .5, .75, .9, .999, 1.01]
COMP = str.maketrans("ACGT", "TGCA")
DOMINANT_MIN = 80.0
# unclassified TE codes are not a superfamily
TE_SKIP = {"XXX", "DXX", "RXX"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_gff(path, keep):
    d = pd.read_csv(path, sep="\t", comment="#", header=None, names=GFF_COLS,
                    usecols=["chrom", "type", "start", "end", "strand", "info"],
                    dtype={"chrom": str})
    return d[d["type"].isin(keep)].rename(columns={"type": "kind"})


def merge_intervals(chrom, start, end):
    """{chrom: (starts, ends)}, sorted and merged, 1-based inclusive."""
    d = pd.DataFrame({"chrom": chrom, "start": start, "end": end}).sort_values(["chrom", "start"])
    out = {}
    for c, g in d.groupby("chrom", sort=False):
        s, e = g["start"].to_numpy(), np.maximum.accumulate(g["end"].to_numpy())
        new = np.r_[True, s[1:] > e[:-1] + 1]
        last = np.r_[new[1:], True]
        out[c] = (s[new], e[last])
    return out


def overlaps_any(merged, chrom, start, end):
    """Boolean per window: does [start, end] touch a merged interval on chrom?"""
    start, end = np.asarray(start), np.asarray(end)
    if chrom not in merged:
        return np.zeros(len(start), bool)
    s, e = merged[chrom]
    i = np.searchsorted(e, start)
    return (i < len(s)) & (s[np.minimum(i, len(s) - 1)] <= end)


def covered_bp(merged, chrom, start, end):
    """Bases of [start, end] covered by the merged intervals on chrom."""
    start, end = np.asarray(start), np.asarray(end)
    if chrom not in merged:
        return np.zeros(len(start), int)
    s, e = merged[chrom]
    cum = np.r_[0, np.cumsum(e - s + 1)]

    def upto(x):
        j = np.searchsorted(e, x)
        part = np.maximum(0, x - s[np.minimum(j, len(s) - 1)] + 1)
        part[j >= len(s)] = 0
        return cum[j] + part
    return upto(end) - upto(start - 1)


class Claimed:
    """Bases already taken by a corpus window, per chromosome."""

    def __init__(self):
        self.s = {}

    def add_many(self, chrom, starts):
        self.s.setdefault(chrom, []).extend(int(x) for x in starts)
        self.s[chrom].sort()

    def free(self, chrom, start):
        a = self.s.get(chrom, [])
        i = bisect.bisect_left(a, start - L + 1)
        return i >= len(a) or a[i] > start + L - 1

    def add(self, chrom, start):
        bisect.insort(self.s.setdefault(chrom, []), start)


class Windows:
    """Sequence fetching with the N and duplicate rules applied."""

    def __init__(self, fasta, seen):
        self.fa = Fasta(fasta, as_raw=True, sequence_always_upper=True)
        self.seen = seen
        self.rejected = {"non_acgt": 0, "duplicate": 0}

    def seq(self, chrom, start, strand):
        s = self.fa[chrom][start - 1:start - 1 + L]
        return s.translate(COMP)[::-1] if strand == "-" else s

    def accept(self, chrom, start, strand):
        s = self.seq(chrom, start, strand)
        if len(s) != L or len(s) != sum(s.count(c) for c in "ACGT"):
            self.rejected["non_acgt"] += 1
            return None
        h = hashlib.sha256(s.encode()).hexdigest()
        if h in self.seen:
            self.rejected["duplicate"] += 1
            return None
        self.seen.add(h)
        return s


def row(wid, gene, group, cls, chrom, strand, tss, start, seq, note=""):
    gc = (seq.count("G") + seq.count("C")) / L
    return dict(window_id=wid, gene_id=gene, group_id=group, region_class=cls, chrom=chrom,
                strand=strand, tss=tss, start=start, end=start + L - 1, gc=round(gc, 4),
                note=note, seq=seq)


def claim_in_order(cands, n, win, claimed, forbid):
    """Walk candidates in the given order and keep the first n that pass every rule."""
    out = []
    drop = {"forbidden": 0, "overlap": 0}
    for c in cands:
        if len(out) >= n:
            break
        ch, st, strand = c["chrom"], c["start"], c["strand"]
        if overlaps_any(forbid, ch, [st], [st + L - 1])[0]:
            drop["forbidden"] += 1
            continue
        if not claimed.free(ch, st):
            drop["overlap"] += 1
            continue
        s = win.accept(ch, st, strand)
        if s is None:
            continue
        claimed.add(ch, st)
        out.append(row(c["wid"], c["gene"], c["group"], c["cls"], ch, strand, c["tss"], st, s,
                       c.get("note", "")))
    return out, drop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hc-gff", required=True)
    ap.add_argument("--lc-gff", required=True)
    ap.add_argument("--te-gff", required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--promoters", required=True)
    ap.add_argument("--promoter-windows", required=True)
    ap.add_argument("--exclusion", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n-intron", type=int, default=12000, help="cap; all qualifying introns if fewer")
    ap.add_argument("--n-utr5", type=int, default=10000)
    ap.add_argument("--n-splice", type=int, default=10000)
    ap.add_argument("--n-intergenic", type=int, default=15000)
    ap.add_argument("--n-repeat", type=int, default=15000)
    ap.add_argument("--gene-pad", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    prom = pd.read_csv(a.promoters, sep="\t", dtype={"chrom": str})
    excl = pd.read_csv(a.exclusion, sep="\t")
    leak = set(excl.loc[excl["reason"].isin(["evaluation_gene", "group_of_evaluation_gene"]), "gene_id"])
    excluded = set(excl["gene_id"])
    group_of = prom.set_index("gene_id")["group_id"].to_dict()

    hc = read_gff(a.hc_gff, {"gene", "mRNA", "exon", "five_prime_UTR"})
    genes = hc[hc["kind"] == "gene"].copy()
    genes["gene_id"] = genes["info"].str.extract(r"ID=([^;]+)", expand=False)
    genes["tss"] = np.where(genes["strand"] == "+", genes["start"], genes["end"])
    corpus_genes = genes[~genes["gene_id"].isin(excluded)].set_index("gene_id")
    print(f"HC genes {len(genes)}; corpus genes {len(corpus_genes)}", flush=True)

    # evaluation genes and their group members: body plus flanks can never be used
    g = genes[genes["gene_id"].isin(leak)]
    forbid = merge_intervals(g["chrom"].to_numpy(), (g["start"] - L).to_numpy(), (g["end"] + L).to_numpy())

    # gene-free space: all HC and LC gene bodies, padded
    lc = read_gff(a.lc_gff, {"gene"})
    allg = pd.concat([genes[["chrom", "start", "end"]], lc[["chrom", "start", "end"]]])
    near_gene = merge_intervals(allg["chrom"].to_numpy(), (allg["start"] - a.gene_pad).to_numpy(),
                                (allg["end"] + a.gene_pad).to_numpy())

    # .1 transcript structure of corpus genes
    tx = hc[hc["kind"].isin(["exon", "five_prime_UTR"])].copy()
    tx["tid"] = tx["info"].str.extract(r"Parent=([^;]+)", expand=False)
    tx = tx[tx["tid"].str.endswith(".1", na=False)]
    tx["gene_id"] = tx["tid"].str[:-2]
    tx = tx[tx["gene_id"].isin(corpus_genes.index)]
    ex = tx[tx["kind"] == "exon"].sort_values(["tid", "start"])
    ex = ex.assign(prev_end=ex.groupby("tid")["end"].shift())
    intr = ex[ex["prev_end"].notna()].copy()
    intr["i_start"] = intr["prev_end"].astype(int) + 1
    intr["i_end"] = intr["start"] - 1
    intr["ilen"] = intr["i_end"] - intr["i_start"] + 1
    with_utr = tx.loc[tx["kind"] == "five_prime_UTR", "gene_id"].unique()

    # promoter component occupies its bases and its sequences
    claimed = Claimed()
    seen = set()
    seq, chunks = None, []
    with open(a.promoter_windows) as f:
        for line in f:
            if line.startswith(">"):
                if seq is not None:
                    seen.add(hashlib.sha256("".join(chunks).encode()).hexdigest())
                m = re.match(r">\S+ (\S+):(\d+)-(\d+)", line)
                claimed.add_many(m.group(1), [int(m.group(2))])
                seq, chunks = True, []
            else:
                chunks.append(line.strip())
    seen.add(hashlib.sha256("".join(chunks).encode()).hexdigest())
    print(f"promoter windows registered: {len(seen)} distinct sequences", flush=True)
    win = Windows(a.fasta, seen)
    chrom_len = {c: len(win.fa[c]) for c in win.fa.keys()}
    placed = [c for c in chrom_len if c != "chrUn"]

    def gene_cand(gid, cls, wid, start, note=""):
        r = corpus_genes.loc[gid]
        return dict(wid=wid, gene=gid, group=group_of.get(gid, gid), cls=cls, chrom=r["chrom"],
                    strand=r["strand"], tss=int(r["tss"]), start=int(start), note=note)

    result, report = [], {"classes": {}}

    # intron
    q = intr[intr["ilen"] >= L]
    q = q.iloc[rng.permutation(len(q))]
    cands = [gene_cand(r.gene_id, "intron", f"intron:{r.tid}:{r.i_start}",
                       r.i_start + (r.ilen - L) // 2) for r in q.itertuples()]
    got, drop = claim_in_order(cands, a.n_intron, win, claimed, forbid)
    result += got
    report["classes"]["intron"] = dict(candidates=len(cands), kept=len(got), genes=len({x["gene_id"] for x in got}), **drop)
    print("intron", report["classes"]["intron"], flush=True)

    # utr5_tss
    ids = rng.permutation(with_utr)
    cands = []
    for gid in ids:
        r = corpus_genes.loc[gid]
        st = int(r["tss"]) if r["strand"] == "+" else int(r["tss"]) - L + 1
        if st >= 1 and st + L - 1 <= chrom_len[r["chrom"]]:
            cands.append(gene_cand(gid, "utr5_tss", f"utr5:{gid}", st))
    got, drop = claim_in_order(cands, a.n_utr5, win, claimed, forbid)
    result += got
    report["classes"]["utr5_tss"] = dict(candidates=len(cands), kept=len(got), genes=len({x["gene_id"] for x in got}), **drop)
    print("utr5_tss", report["classes"]["utr5_tss"], flush=True)

    # splice: one boundary per gene; a later candidate of the same gene can fill in if the first fails
    by_gene = {gid: d for gid, d in intr.groupby("gene_id")}
    cands = []
    for gid in rng.permutation(list(by_gene)):
        d = by_gene[gid]
        r = corpus_genes.loc[gid]
        for k in rng.permutation(len(d)):
            i = d.iloc[k]
            left = rng.random() < .5
            b = int(i["i_start"]) if left else int(i["i_end"])
            st = b - L // 2 if left else b - L // 2 + 1
            donor = left == (r["strand"] == "+")
            if st >= 1 and st + L - 1 <= chrom_len[r["chrom"]]:
                cands.append(gene_cand(gid, "splice", f"splice:{gid}:{b}", st, "donor" if donor else "acceptor"))
    kept, drop = [], {"forbidden": 0, "overlap": 0}
    used = set()
    for c in cands:
        if len(kept) >= a.n_splice:
            break
        if c["gene"] in used:
            continue
        got, d = claim_in_order([c], 1, win, claimed, forbid)
        for k2 in drop:
            drop[k2] += d[k2]
        if got:
            kept += got
            used.add(c["gene"])
    result += kept
    report["classes"]["splice"] = dict(candidates_genes=len(by_gene), kept=len(kept),
                                       donor=sum(x["note"] == "donor" for x in kept),
                                       acceptor=sum(x["note"] == "acceptor" for x in kept), **drop)
    print("splice", report["classes"]["splice"], flush=True)

    # TE annotation
    te_rows, reg = [], []
    for ch in pd.read_csv(a.te_gff, sep="\t", comment="#", header=None, names=GFF_COLS,
                          usecols=["chrom", "type", "start", "end", "strand", "info"],
                          dtype={"chrom": str}, chunksize=500_000):
        te_rows.append(ch[["chrom", "start", "end"]])
        r = ch[(ch["type"] == "repeat_region") & (ch["end"] - ch["start"] + 1 >= L)]
        if len(r):
            r = r.assign(compo=r["info"].str.extract(r"compo=([^;]+)", expand=False))
            reg.append(r[["chrom", "start", "end", "strand", "compo"]])
    te_all = pd.concat(te_rows)
    te_m = merge_intervals(te_all["chrom"].to_numpy(), te_all["start"].to_numpy(), te_all["end"].to_numpy())
    del te_rows, te_all
    print(f"TE intervals merged: {sum(int((e - s + 1).sum()) for s, e in te_m.values())} bp", flush=True)

    # intergenic, matched to promoters on subgenome x TE bin x GC bin
    pt = np.digitize(prom["te_frac"], TE_BINS) - 1
    gc_edges = np.quantile(prom["gc"], np.linspace(0, 1, 11))
    gc_edges[0], gc_edges[-1] = 0, 1.0001
    pg = np.digitize(prom["gc"], gc_edges) - 1
    ps = prom["chrom"].str[-1].to_numpy()
    placed_mask = prom["chrom"] != "chrUn"
    cells = pd.Series(list(zip(ps[placed_mask], pt[placed_mask], pg[placed_mask]))).value_counts(normalize=True)
    want = {c: int(round(f * a.n_intergenic)) for c, f in cells.items()}

    tot = sum(chrom_len[c] for c in placed)
    pool = []
    per_te_cap = {}
    for sg_te, f in pd.Series(list(zip(ps[placed_mask], pt[placed_mask]))).value_counts(normalize=True).items():
        per_te_cap[sg_te] = int(f * a.n_intergenic * 12) + 200
    n_draw = 6_000_000
    for c in placed:
        n = int(n_draw * chrom_len[c] / tot)
        st = np.unique(rng.integers(1, chrom_len[c] - L, n))
        st = st[~overlaps_any(near_gene, c, st, st + L - 1) & ~overlaps_any(forbid, c, st, st + L - 1)]
        tb = np.digitize(covered_bp(te_m, c, st, st + L - 1) / L, TE_BINS) - 1
        pool.append(pd.DataFrame({"chrom": c, "start": st, "sg": c[-1], "tb": tb}))
    pool = pd.concat(pool, ignore_index=True)
    pool = pool.iloc[rng.permutation(len(pool))].reset_index(drop=True)
    pool["rank"] = pool.groupby(["sg", "tb"]).cumcount()
    cap = pool.apply(lambda r: per_te_cap.get((r["sg"], r["tb"]), 0), axis=1) if False else \
        pd.Series([per_te_cap.get(k, 0) for k in zip(pool["sg"], pool["tb"])])
    pool = pool[pool["rank"] < cap].reset_index(drop=True)
    print(f"intergenic candidate pool: {len(pool)}", flush=True)

    kept = []
    filled = dict.fromkeys(want, 0)
    drop = {"overlap": 0, "cell_full": 0}
    for r in pool.itertuples():
        if not claimed.free(r.chrom, r.start):
            drop["overlap"] += 1
            continue
        s = win.seq(r.chrom, r.start, "+")
        if len(s) != L or len(s) != sum(s.count(ch) for ch in "ACGT"):
            win.rejected["non_acgt"] += 1
            continue
        gcv = (s.count("G") + s.count("C")) / L
        cell = (r.sg, r.tb, int(np.digitize(gcv, gc_edges) - 1))
        if filled.get(cell, 0) >= want.get(cell, 0):
            drop["cell_full"] += 1
            continue
        s2 = win.accept(r.chrom, r.start, "+")
        if s2 is None:
            continue
        filled[cell] += 1
        claimed.add(r.chrom, r.start)
        wid = f"intergenic:{r.chrom}:{r.start}"
        kept.append(row(wid, wid, wid, "intergenic", r.chrom, "+", 0, r.start, s2))
    result += kept
    short = {str(k): want[k] - filled[k] for k in want if filled[k] < want[k]}
    igt = pd.DataFrame(kept)
    te_ig = np.concatenate([covered_bp(te_m, c, d["start"].to_numpy(), d["end"].to_numpy()) / L
                            for c, d in igt.groupby("chrom")])
    smd = lambda x, y: float((np.mean(x) - np.mean(y)) / np.sqrt((np.var(x) + np.var(y)) / 2))
    report["classes"]["intergenic"] = dict(
        target=a.n_intergenic, kept=len(kept), pool=len(pool), shortfall_total=sum(short.values()),
        shortfall_cells=len(short), smd_gc=smd(igt["gc"], prom["gc"]), smd_te=smd(te_ig, prom["te_frac"]),
        mean_gc=float(igt["gc"].mean()), mean_te=float(te_ig.mean()),
        promoter_mean_gc=float(prom["gc"].mean()), promoter_mean_te=float(prom["te_frac"].mean()), **drop)
    print("intergenic", report["classes"]["intergenic"], flush=True)

    # repeat, stratified by dominant superfamily
    regs = pd.concat(reg, ignore_index=True)
    regs = regs[regs["chrom"].isin(placed) & regs["compo"].notna()]
    parts = regs["compo"].str.findall(r"([A-Za-z]{3})_\S*\s+([\d.]+)")

    def dominant(p):
        tot_ = {}
        for fam, pct in p:
            tot_[fam] = tot_.get(fam, 0.0) + float(pct)
        if not tot_:
            return None
        fam = max(tot_, key=tot_.get)
        return fam if tot_[fam] >= DOMINANT_MIN and fam not in TE_SKIP else None
    regs["sf"] = parts.map(dominant)
    regs = regs[regs["sf"].notna()]
    avail = regs["sf"].value_counts()
    print("repeat elements by superfamily:", avail.to_dict(), flush=True)
    regs = regs.iloc[rng.permutation(len(regs))].reset_index(drop=True)
    offs = rng.random(len(regs))
    regs["wstart"] = regs["start"] + (offs * (regs["end"] - regs["start"] - L + 2)).astype(int)
    ok = np.zeros(len(regs), bool)
    for c, d in regs.groupby("chrom"):
        st = d["wstart"].to_numpy()
        ok[d.index] = ~overlaps_any(near_gene, c, st, st + L - 1) & ~overlaps_any(forbid, c, st, st + L - 1)
    regs = regs[ok]
    sfs = list(regs["sf"].value_counts().index)
    queues = {s: iter(regs[regs["sf"] == s].itertuples()) for s in sfs}
    counts = dict.fromkeys(sfs, 0)
    active = sorted(sfs)
    kept = []
    while len(kept) < a.n_repeat and active:  # round robin: equal allocation until a family runs out
        for s in list(active):
            if len(kept) >= a.n_repeat:
                break
            for r in queues[s]:
                strand = r.strand if r.strand in "+-" else "+"
                if not claimed.free(r.chrom, r.wstart):
                    continue
                q = win.accept(r.chrom, r.wstart, strand)
                if q is None:
                    continue
                claimed.add(r.chrom, r.wstart)
                wid = f"repeat:{s}:{r.chrom}:{r.wstart}"
                kept.append(row(wid, wid, wid, "repeat", r.chrom, strand, 0, r.wstart, q, s))
                counts[s] += 1
                break
            else:
                active.remove(s)
    result += kept
    report["classes"]["repeat"] = dict(target=a.n_repeat, kept=len(kept), elements_eligible=int(len(regs)),
                                       by_superfamily=counts, elements_by_superfamily={k: int(v) for k, v in avail.items()})
    print("repeat", report["classes"]["repeat"], flush=True)

    # final checks and output
    d = pd.DataFrame(result).sort_values(["region_class", "chrom", "start"]).reset_index(drop=True)
    if d["window_id"].duplicated().any():
        raise SystemExit("duplicate window ids")
    genic = d[~d["region_class"].isin(["intergenic", "repeat"])]
    if genic["gene_id"].isin(excluded).any():
        raise SystemExit("a genic window belongs to an excluded gene")
    bad = sum(int(overlaps_any(forbid, c, x["start"], x["end"]).sum()) for c, x in d.groupby("chrom"))
    if bad:
        raise SystemExit(f"{bad} windows touch an evaluation gene region")
    for c, x in d.groupby("chrom"):
        st = np.sort(x["start"].to_numpy())
        if (np.diff(st) < L).any():
            raise SystemExit(f"overlapping new windows on {c}")

    fa_path = os.path.join(a.outdir, "corpus_regions_L1002.fa")
    h = hashlib.sha256()
    with open(fa_path, "w", newline="\n") as out:
        for r in d.itertuples():
            out.write(f">{r.window_id} {r.chrom}:{r.start}-{r.end}({r.strand}) tss={r.tss} len={L}\n")
            for i in range(0, L, 60):
                out.write(r.seq[i:i + 60] + "\n")
            h.update(r.seq.encode())
    tsv = os.path.join(a.outdir, "corpus_regions_L1002.tsv")
    d.drop(columns="seq").to_csv(tsv, sep="\t", index=False)

    report.update(
        seed=a.seed, length_bp=L, gene_pad=a.gene_pad, n_windows=len(d), n_tokens=len(d) * L // 6,
        rejected=win.rejected, windows_content_sha256=h.hexdigest(), manifest_sha256=sha256(tsv),
        inputs={os.path.basename(p): sha256(p) for p in (a.hc_gff, a.lc_gff, a.promoters, a.exclusion)},
        by_class=d["region_class"].value_counts().to_dict(),
        by_subgenome=d["chrom"].str[-1].value_counts().to_dict(),
        gc_by_class=d.groupby("region_class")["gc"].mean().round(4).to_dict())
    with open(os.path.join(a.outdir, "build_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: report[k] for k in ("n_windows", "n_tokens", "by_class", "by_subgenome", "gc_by_class", "rejected", "windows_content_sha256")},
                     indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


if __name__ == "__main__":
    main()
