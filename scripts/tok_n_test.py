#!/usr/bin/env python3
"""
Show how AgroNT's tokeniser handles N. Runs on CPU, offline, using the
local model folder; no model weights are loaded.

Usage:
  python scripts/tok_n_test.py --model models/agro-nt-1b
"""
import argparse
import random

from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.model)

print(f"tokeniser class: {type(tok).__name__}; vocab {len(tok)}")
print(f"'N' in vocab: {'N' in tok.get_vocab()}  "
      f"unk token: {tok.unk_token!r} id {tok.unk_token_id}")
print(f"any 6-mer containing N in vocab: "
      f"{any(len(k) == 6 and 'N' in k for k in tok.get_vocab())}")

random.seed(0)
s = "".join(random.choice("ACGT") for _ in range(1002))
cases = {
    "clean 1002": s,
    "NN at positions 1-2": "NN" + s[2:],
    "single N at position 500": s[:499] + "N" + s[500:],
    "run of 10 N at 301-310": s[:300] + "N" * 10 + s[310:],
    "clean 1000 (old length)": s[:1000],
}
for name, x in cases.items():
    enc = tok(x)
    toks = tok.convert_ids_to_tokens(enc["input_ids"])
    odd = [(i, t) for i, t in enumerate(toks) if len(t) != 6]
    print(f"\n{name}: {len(toks)} tokens incl. specials")
    print(f"  first 3: {toks[:3]}  last 3: {toks[-3:]}")
    print(f"  tokens that are not 6-mers ({len(odd)}): {odd[:16]}")
    if "N" in x:
        # does the 6-mer frame after the N line up with the clean sequence?
        clean = set(tok.convert_ids_to_tokens(tok(s)["input_ids"])[-20:])
        tail = toks[-20:]
        print(f"  last 20 tokens shared with clean window: "
              f"{sum(t in clean for t in tail)}/20 "
              f"(20/20 = frame preserved, low = frame shifted)")
