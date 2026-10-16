#!/usr/bin/env python3
"""
Extract promoter windows from the reference FASTA, oriented 5'->3' on the
gene's own strand, ending at the base immediately upstream of the TSS.

  + strand: reference[tss-L .. tss-1]
  - strand: reverse_complement(reference[tss+1 .. tss+L])

So in every output sequence, position 1 is the distal end and position L is
the base adjacent to the transcription start site. Tokenisation runs left to
right from the distal end.

Runs on any machine with the FASTA; needs no GPU.

Usage:
  python extract_windows.py --manifest data/benchmark_sample.tsv \
      --fasta /path/161010_Chinese_Spring_v1.0_pseudomolecules.fasta \
      --length 1000 --out data/benchmark_windows.fa
"""
import argparse
import hashlib

import pandas as pd
from pyfaidx import Fasta

COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(s):
    return s.translate(COMP)[::-1]


def window(fa, chrom, strand, tss, L):
    if strand == "+":
        start, end = tss - L, tss - 1
        if start < 1:
            return None, (start, end)
        seq = fa[chrom][start - 1:end]
    elif strand == "-":
        start, end = tss + 1, tss + L
        if end > len(fa[chrom]):
            return None, (start, end)
        seq = revcomp(fa[chrom][start - 1:end])
    else:
        raise ValueError(f"bad strand {strand!r}")
    return seq.upper(), (start, end)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--length", type=int, default=1002)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    m = pd.read_csv(a.manifest, sep="\t")
    fa = Fasta(a.fasta, as_raw=True, sequence_always_upper=True)
    missing = sorted(set(m["chrom"]) - set(fa.keys()))
    if missing:
        raise SystemExit(f"chromosomes not in FASTA: {missing[:5]} "
                         f"(FASTA has e.g. {list(fa.keys())[:3]})")

    n_ok = n_skip = n_mismatch = n_nonacgt = 0
    bad_id = []
    h = hashlib.sha256()
    with open(a.out, "w") as out:
        for r in m.itertuples(index=False):
            seq, (s, e) = window(fa, r.chrom, r.strand, int(r.tss), a.length)
            if seq is None or len(seq) != a.length:
                n_skip += 1
                continue

            d = a.length - 1000
            expect = ((r.prom_start - d, r.prom_end) if r.strand == "+"
                      else (r.prom_start, r.prom_end + d))
            if (s, e) != expect:
                n_mismatch += 1
            if len(seq) - sum(seq.count(c) for c in "ACGT"):
                n_nonacgt += 1
                bad_id.append(r.gene_id)
            header = (f">{r.gene_id} {r.chrom}:{s}-{e}({r.strand}) "
                      f"tss={r.tss} len={a.length}")
            out.write(header + "\n")
            for i in range(0, len(seq), 60):
                out.write(seq[i:i + 60] + "\n")
            h.update(seq.encode())
            n_ok += 1

    print(f"wrote {n_ok} windows to {a.out}; skipped {n_skip} at contig edges")
    print(f"coordinate mismatches against manifest: {n_mismatch}")
    print(f"windows with non-ACGT bases: {n_nonacgt} {bad_id[:10]}")
    print(f"sequence-content sha256: {h.hexdigest()[:16]}…")


if __name__ == "__main__":
    main()
