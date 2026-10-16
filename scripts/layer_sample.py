#!/usr/bin/env python3
"""
Extract per-window summaries of AgroNT activations at the candidate layers,
for the layer-selection probe.

For each window and each layer, two summaries over real sequence tokens
(CLS, EOS and padding excluded):
  mean  average activation across positions
  max   per-dimension maximum across positions (local presence)
Layer 0 is the embedding output before any transformer block, included as a
no-context reference.

Output is small (windows x 1,500 floats per layer per summary), so it can be
copied to a laptop for the CPU probe step.

Usage (lab GPU):
  python layer_sample.py --model models/agro-nt-1b --windows data/probe_windows.fa \
      --layers 0,4,12,20,30,40 --out data/probe/layer_summaries.npz
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark import DTYPES, encode, load, read_fasta, real_token_mask  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--layers", default="0,4,12,20,30,40")
    ap.add_argument("--batch", type=int, default=16,
                    help="use the batch size chosen from the benchmark sweep")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--max-seqs", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if a.device == "auto" else a.device)
    if a.dtype == "auto":
        a.dtype = ("bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported()
                   else "fp32")
    layers = [int(x) for x in a.layers.split(",")]
    ids, seqs = read_fasta(a.windows, a.max_seqs)
    tok, model = load(a.model, a.revision, device, DTYPES[a.dtype])
    n_layers = model.config.num_hidden_layers
    if max(layers) > n_layers or min(layers) < 0:
        raise SystemExit(f"layers must be within 0..{n_layers}")
    H = model.config.hidden_size
    print(f"{len(seqs)} windows, layers {layers}, device {device}, dtype {a.dtype}")

    out = {f"{s}_L{k}": np.zeros((len(seqs), H), dtype=np.float32)
           for k in layers for s in ("mean", "max")}
    n_tokens = np.zeros(len(seqs), dtype=np.int32)
    t0 = time.perf_counter()
    for b in range(0, len(seqs), a.batch):
        enc = encode(tok, seqs[b:b + a.batch], device)
        keep = real_token_mask(tok, enc)
        hs = model(**enc, output_hidden_states=True).hidden_states
        for k in layers:
            x = hs[k].float()
            m = keep.unsqueeze(-1)
            cnt = m.sum(dim=1).clamp_min(1)
            out[f"mean_L{k}"][b:b + a.batch] = ((x * m).sum(dim=1) / cnt).cpu().numpy()
            out[f"max_L{k}"][b:b + a.batch] = (
                x.masked_fill(~m, float("-inf")).max(dim=1).values.cpu().numpy())
        n_tokens[b:b + a.batch] = keep.sum(dim=1).cpu().numpy()
        if (b // a.batch) % 20 == 0:
            print(f"  {min(b + a.batch, len(seqs))}/{len(seqs)}")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, ids=np.array(ids), layers=np.array(layers),
                        n_tokens=n_tokens, **out)
    dt = time.perf_counter() - t0
    print(f"wrote {a.out} in {dt:.0f}s; real tokens per window "
          f"min {n_tokens.min()} median {int(np.median(n_tokens))} max {n_tokens.max()}")


if __name__ == "__main__":
    main()
