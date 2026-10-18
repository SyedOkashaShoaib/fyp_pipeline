#!/usr/bin/env python3
"""
Audit extracted windows for non-ACGT bases, and optionally compare against
the n_frac column of the covariates file.

Per window it counts N and any other non-ACGT letter (IUPAC codes such as
R, Y), and records where they sit (1-based; position 1 = distal end, the
last position = base adjacent to the TSS).

Usage:
  # inspect the probe windows
  python scripts/window_audit.py --fasta data/probe_windows.fa

  # check cov_dry.tsv n_frac against the real 1,000 bp sequence
  python scripts/window_audit.py --fasta audit/corpus_L1000.fa --covariates cov_dry.tsv

  # how many corpus windows contain N at 1,002 bp
  python scripts/window_audit.py --fasta audit/corpus_L1002.fa --out audit/corpus_L1002_nfrac.tsv
"""
import argparse

import pandas as pd


def read_fasta(path):
    name, cur = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(cur)
                name, cur = line[1:].split()[0], []
            elif line:
                cur.append(line.upper())
    if name is not None:
        yield name, "".join(cur)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--covariates", default=None,
                    help="cov_dry.tsv; compares its n_frac with the sequence")
    ap.add_argument("--out", default=None, help="optional per-window TSV")
    a = ap.parse_args()

    rows = []
    for gid, s in read_fasta(a.fasta):
        bad = [i + 1 for i, c in enumerate(s) if c not in "ACGT"]
        rows.append(dict(
            gene_id=gid, length=len(s),
            n_N=s.count("N"),
            n_other=len(bad) - s.count("N"),
            frac_nonacgt=len(bad) / len(s),
            first_bad=bad[0] if bad else 0,
            last_bad=bad[-1] if bad else 0,
            letters="".join(sorted({s[i - 1] for i in bad}))))
    d = pd.DataFrame(rows)
    L = sorted(int(x) for x in d["length"].unique())
    print(f"{len(d)} windows; lengths {L}")

    has = d["frac_nonacgt"] > 0
    print(f"windows with any non-ACGT: {has.sum()} ({has.mean():.2%})")
    print(f"  with a non-N letter (IUPAC etc.): {(d['n_other'] > 0).sum()}")
    bands = pd.cut(d["frac_nonacgt"], [-1, 0, 0.01, 0.05, 0.1, 1.0],
                   labels=["0", "(0,1%]", "(1,5%]", "(5,10%]", ">10%"])
    print("by non-ACGT fraction:", bands.value_counts().sort_index().to_dict())

    if 0 < has.sum() <= 20:
        print("\nwindows with non-ACGT bases:")
        print(d[has][["gene_id", "length", "n_N", "n_other", "first_bad",
                      "last_bad", "letters"]].to_string(index=False))

    if a.covariates:
        cov = pd.read_csv(a.covariates, sep="\t").set_index("gene_id")["n_frac"]
        d["cov_n_frac"] = d["gene_id"].map(cov)
        miss = d["cov_n_frac"].isna().sum()
        # n_frac may be rounded; allow half a unit of the 3rd decimal
        tol = 0.0005 + 1e-9
        d["frac_N"] = d["n_N"] / d["length"]
        for col in ("frac_N", "frac_nonacgt"):
            diff = (d[col] - d["cov_n_frac"]).abs()
            print(f"\ncov n_frac vs {col}: max |diff| {diff.max():.6f}; "
                  f"windows beyond rounding tolerance: {(diff > tol).sum()}")
        print(f"genes missing from covariates: {miss}")
        zero_cov = d["cov_n_frac"] == 0
        print(f"cov says n_frac == 0 but sequence has non-ACGT: "
              f"{(zero_cov & has).sum()}")

    if a.out:
        d.to_csv(a.out, sep="\t", index=False, lineterminator="\n")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
