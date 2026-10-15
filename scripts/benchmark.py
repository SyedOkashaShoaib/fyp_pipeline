#!/usr/bin/env python3
"""
AgroNT extraction benchmark (deliverable D3, gate G2).

Stages, run individually or all together:
  smoke        load model and tokenizer, print and assert the configuration
  tokens       token-count distribution and special-token layout
  sweep        peak memory and throughput across batch sizes
  precision    fp32 vs reduced precision, per candidate layer
  hooks        forward hook on encoder.layer[k-1] must equal hidden_states[k]
  determinism  same batch twice gives the same output
  shards       write activations as checksummed, atomically renamed shards
               (rerun after interruption: it resumes and verifies)
  project      wall-clock and storage projection for the full corpus

Designed to run on CPU with a handful of sequences as a dry run at home,
then unchanged on the lab GPU.

Home dry run (CPU, a few sequences, no assertions failing the run):
  python benchmark.py --model /path/agro-nt --windows data/benchmark_windows.fa \
      --device cpu --max-seqs 8 --batch-sizes 1,2 --stages smoke,tokens,hooks

Lab run:
  python benchmark.py --model /path/agro-nt --windows data/benchmark_windows.fa \
      --stages all --outdir bench_out
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np
import torch

EXPECTED = {"hidden_size": 1500, "num_hidden_layers": 40,
            "num_attention_heads": 20, "vocab_size": 4105,
            "max_position_embeddings": 1026}
PINNED_REVISION = "9be6ae0d285053466d76ec50b042b8628065efd0"
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


# --------------------------------------------------------------------- utils

def log(msg=""):
    print(msg, flush=True)


def read_fasta(path, limit=None):
    ids, seqs, cur, name = [], [], [], None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if name is not None:
                    ids.append(name)
                    seqs.append("".join(cur))
                    if limit and len(ids) >= limit:
                        return ids, seqs
                name, cur = line[1:].split()[0], []
            elif line:
                cur.append(line.upper())
    if name is not None and (not limit or len(ids) < limit):
        ids.append(name)
        seqs.append("".join(cur))
    return ids, seqs


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def environment():
    env = {"python": sys.version.split()[0], "torch": torch.__version__,
           "platform": platform.platform()}
    try:
        import transformers
        env["transformers"] = transformers.__version__
    except Exception:
        pass
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        env.update(gpu=p.name, gpu_memory_gb=round(p.total_memory / 2**30, 2),
                   cuda=torch.version.cuda,
                   bf16_supported=torch.cuda.is_bf16_supported())
    return env


# --------------------------------------------------------------- model setup

def load(model_path, revision, device, dtype):
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    kw = {"revision": revision} if revision else {}
    tok = AutoTokenizer.from_pretrained(model_path, **kw)
    model = AutoModelForMaskedLM.from_pretrained(model_path, torch_dtype=dtype, **kw)
    model.eval().to(device)
    return tok, model


def encoder_layers(model):
    """The list of transformer blocks, located without assuming a class name."""
    for path in ("esm.encoder.layer", "encoder.layer", "base_model.encoder.layer"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj, path
        except AttributeError:
            continue
    raise RuntimeError("could not locate encoder blocks; inspect print(model)")


def encode(tok, seqs, device):
    batch = tok(seqs, return_tensors="pt", padding=True)
    return {k: v.to(device) for k, v in batch.items()}


def real_token_mask(tok, batch):
    """True at positions holding sequence tokens: not padding, CLS or EOS."""
    ids = batch["input_ids"]
    keep = batch["attention_mask"].bool()
    for special in (tok.cls_token_id, tok.eos_token_id, tok.bos_token_id,
                    tok.pad_token_id):
        if special is not None:
            keep &= ids != special
    return keep


@torch.no_grad()
def forward(model, batch):
    return model(**batch, output_hidden_states=True)


# -------------------------------------------------------------------- stages

def stage_smoke(args, tok, model, report):
    cfg = model.config
    got = {k: getattr(cfg, k, None) for k in EXPECTED}
    n_params = sum(p.numel() for p in model.parameters())
    log(f"architecture: {type(model).__name__}  parameters: {n_params/1e6:.1f}M")
    log(f"config: {got}")
    log(f"emb_layer_norm_before={getattr(cfg, 'emb_layer_norm_before', None)}  "
        f"position_embedding_type={getattr(cfg, 'position_embedding_type', None)}")
    bad = {k: (got[k], v) for k, v in EXPECTED.items() if got[k] != v}
    if bad:
        msg = f"configuration differs from the pinned AgroNT values: {bad}"
        if args.strict:
            raise SystemExit(msg)
        log("WARNING: " + msg + "  (expected only when testing with a stand-in model)")
    report["smoke"] = {"config": got, "parameters": n_params,
                       "matches_expected": not bad}


def stage_tokens(args, tok, ids, seqs, report):
    lens, n_count = [], []
    for s in seqs:
        enc = tok(s)["input_ids"]
        lens.append(len(enc))
        n_count.append(s.count("N"))
    lens = np.array(lens)
    first = tok(seqs[0])["input_ids"]
    toks = tok.convert_ids_to_tokens(first)
    log(f"sequence length (bases): {len(seqs[0])}")
    log(f"tokens per window incl. specials: min {lens.min()}  "
        f"median {int(np.median(lens))}  max {lens.max()}")
    log(f"first 4 tokens: {toks[:4]}   last 4 tokens: {toks[-4:]}")
    log(f"CLS present: {tok.cls_token_id in first}   "
        f"EOS present: {tok.eos_token_id is not None and tok.eos_token_id in first}")
    single = [t for t in toks if len(t) == 1 and t.isalpha()]
    log(f"single-nucleotide tokens in window 1: {len(single)} -> {single[:8]}")
    with_n = int((np.array(n_count) > 0).sum())
    log(f"windows containing any N: {with_n} of {len(seqs)}")
    report["tokens"] = {"min": int(lens.min()), "median": float(np.median(lens)),
                        "max": int(lens.max()), "first_tokens": toks[:4],
                        "last_tokens": toks[-4:],
                        "single_nt_tokens_window1": len(single),
                        "windows_with_N": with_n}


def stage_sweep(args, tok, model, seqs, device, report):
    rows = []
    sizes = [int(x) for x in args.batch_sizes.split(",")]
    for bs in sizes:
        if bs > len(seqs):
            break
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            batches = [seqs[i:i + bs] for i in range(0, len(seqs) - bs + 1, bs)]
            warm, timed = batches[:args.warmup], batches[args.warmup:]
            if not timed:
                timed = batches
            for b in warm:
                forward(model, encode(tok, b, device))
            sync(device)
            n_win = n_tok = 0
            t0 = time.perf_counter()
            for b in timed:
                enc = encode(tok, b, device)
                forward(model, enc)
                n_win += len(b)
                n_tok += int(enc["attention_mask"].sum())
            sync(device)
            dt = time.perf_counter() - t0
            peak = (torch.cuda.max_memory_allocated() / 2**30
                    if device.type == "cuda" else float("nan"))
            row = {"batch_size": bs, "windows_per_s": n_win / dt,
                   "tokens_per_s": n_tok / dt, "peak_gb": peak}
            rows.append(row)
            log(f"bs={bs:<4} {row['windows_per_s']:8.1f} win/s  "
                f"{row['tokens_per_s']:10.0f} tok/s  peak {peak:6.2f} GB")
        except torch.cuda.OutOfMemoryError:
            log(f"bs={bs:<4} out of memory - stopping sweep")
            torch.cuda.empty_cache()
            break
    report["sweep"] = rows


def stage_precision(args, tok, model_path, revision, seqs, layers, device, report):
    sub = seqs[:min(len(seqs), args.precision_n)]
    _, m32 = load(model_path, revision, device, torch.float32)
    enc = encode(tok, sub, device)
    keep = real_token_mask(tok, enc)
    ref = forward(m32, enc).hidden_states
    ref = {k: ref[k][keep].float().cpu() for k in layers}
    del m32
    if device.type == "cuda":
        torch.cuda.empty_cache()
    out = {}
    for name in args.reduced.split(","):
        if name == "fp32":
            continue
        _, m = load(model_path, revision, device, DTYPES[name])
        hs = forward(m, enc).hidden_states
        out[name] = {}
        for k in layers:
            x = hs[k][keep].float().cpu()
            finite = bool(torch.isfinite(x).all())
            cos = torch.nn.functional.cosine_similarity(x, ref[k], dim=-1)
            rel = ((x - ref[k]).norm(dim=-1) / ref[k].norm(dim=-1).clamp_min(1e-8))
            out[name][k] = {"finite": finite, "cos_min": float(cos.min()),
                            "cos_mean": float(cos.mean()),
                            "rel_err_mean": float(rel.mean()),
                            "rel_err_max": float(rel.max()),
                            "max_abs_activation": float(ref[k].abs().max())}
            log(f"{name} layer {k:>2}: finite={finite}  cos mean {cos.mean():.5f} "
                f"min {cos.min():.5f}  rel.err mean {rel.mean():.2e} "
                f"max {rel.max():.2e}  |x|max {ref[k].abs().max():.1f}")
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report["precision"] = out


def stage_hooks(args, tok, model, seqs, layers, device, report):
    blocks, path = encoder_layers(model)
    log(f"encoder blocks at model.{path}: {len(blocks)}")
    captured = {}
    handles = []
    for k in layers:
        def hook(_m, _inp, output, k=k):
            captured[k] = (output[0] if isinstance(output, tuple) else output).detach()
        handles.append(blocks[k - 1].register_forward_hook(hook))
    enc = encode(tok, seqs[:min(4, len(seqs))], device)
    hs = forward(model, enc).hidden_states
    for h in handles:
        h.remove()
    log(f"hidden_states returned: {len(hs)} (expected {len(blocks) + 1}: "
        f"embeddings + one per block)")
    res = {}
    for k in layers:
        diff = float((captured[k].float() - hs[k].float()).abs().max())
        same = diff < 1e-5
        res[k] = {"max_abs_diff": diff, "identical": same}
        note = ""
        if not same and k == len(blocks):
            note = "  <- final layer norm applied to the last hidden state"
        log(f"layer {k:>2}: hook vs hidden_states[{k}] max|diff| = {diff:.3e}  "
            f"{'IDENTICAL' if same else 'DIFFERENT'}{note}")
    report["hooks"] = {"n_hidden_states": len(hs), "n_blocks": len(blocks),
                       "block_path": path, "layers": res}


def stage_determinism(args, tok, model, seqs, layers, device, report):
    enc = encode(tok, seqs[:min(8, len(seqs))], device)
    a = forward(model, enc).hidden_states
    b = forward(model, enc).hidden_states
    res = {k: float((a[k].float() - b[k].float()).abs().max()) for k in layers}
    for k, v in res.items():
        log(f"layer {k:>2}: max|run1-run2| = {v:.3e}")
    report["determinism"] = res


def stage_shards(args, tok, model, ids, seqs, layer, device, report):
    sdir = os.path.join(args.outdir, f"shards_L{layer}")
    os.makedirs(sdir, exist_ok=True)
    manifest = os.path.join(sdir, "manifest.jsonl")

    for f in os.listdir(sdir):
        if f.endswith(".tmp"):
            os.remove(os.path.join(sdir, f))
            log(f"removed incomplete shard {f}")

    done, verified = set(), 0
    if os.path.exists(manifest):
        with open(manifest) as f:
            for line in f:
                rec = json.loads(line)
                path = os.path.join(sdir, rec["file"])
                if not os.path.exists(path) or sha256_file(path) != rec["sha256"]:
                    raise SystemExit(f"checksum failure on {rec['file']}; "
                                     "the store is corrupt, do not resume")
                done.update(rec["ids"])
                verified += 1
        log(f"resuming: {verified} shards verified, {len(done)} windows already done")

    todo = [(i, s) for i, s in zip(ids, seqs) if i not in done]
    shard_n = args.shard_windows
    n_shards = len([f for f in os.listdir(sdir) if f.endswith(".npy")])
    t0, written_bytes, wrote = time.perf_counter(), 0, 0

    for start in range(0, len(todo), shard_n):
        chunk = todo[start:start + shard_n]
        acts, offsets, chunk_ids = [], [0], []
        for b in range(0, len(chunk), args.extract_batch):
            part = chunk[b:b + args.extract_batch]
            enc = encode(tok, [s for _, s in part], device)
            keep = real_token_mask(tok, enc)
            hs = forward(model, enc).hidden_states[layer]
            for j in range(len(part)):
                a = hs[j][keep[j]].float().cpu().numpy()
                if np.abs(a).max() > 65000:
                    raise SystemExit("activation exceeds float16 range; "
                                     "store as bfloat16 instead")
                acts.append(a.astype(np.float16))
                offsets.append(offsets[-1] + a.shape[0])
                chunk_ids.append(part[j][0])
        arr = np.concatenate(acts, axis=0)
        name = f"shard_{n_shards:05d}.npy"
        tmp = os.path.join(sdir, name + ".tmp")
        with open(tmp, "wb") as f:
            np.save(f, arr)
            if args.simulate_kill_after is not None and \
                    n_shards >= args.simulate_kill_after:
                f.flush()
                log(f"simulated kill while writing {name}")
                os._exit(9)
            f.flush()
            os.fsync(f.fileno())
        digest = sha256_file(tmp)
        os.replace(tmp, os.path.join(sdir, name))
        rec = {"file": name, "sha256": digest, "ids": chunk_ids,
               "offsets": offsets, "shape": list(arr.shape),
               "dtype": "float16", "layer": layer}
        with open(manifest, "a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
        written_bytes += arr.nbytes
        wrote += len(chunk_ids)
        n_shards += 1

    dt = time.perf_counter() - t0
    total = set()
    with open(manifest) as f:
        for line in f:
            total.update(json.loads(line)["ids"])
    missing = set(ids) - total
    log(f"shards complete: {n_shards}; windows in store {len(total)} of {len(ids)}; "
        f"missing {len(missing)}")
    if wrote:
        log(f"this run wrote {wrote} windows, {written_bytes/2**30:.3f} GB "
            f"in {dt:.1f}s -> {written_bytes/2**30/dt*3600:.1f} GB/h")
    report["shards"] = {"layer": layer, "shards": n_shards,
                        "windows_in_store": len(total), "missing": len(missing),
                        "gb_written_this_run": written_bytes / 2**30,
                        "seconds": dt}


def stage_project(args, report):
    sweep = report.get("sweep") or []
    tokens = report.get("tokens") or {}
    if not sweep or not tokens:
        log("projection needs the sweep and tokens stages")
        return
    best = max(sweep, key=lambda r: r["windows_per_s"])
    real_tokens = tokens["median"] - 1  # CLS removed; adjust if EOS present
    if "[EOS]" in str(tokens.get("last_tokens")) or "<eos>" in str(tokens.get("last_tokens")):
        real_tokens -= 1
    mb = real_tokens * EXPECTED["hidden_size"] * 2 / 1e6
    proj = {}
    for n in [int(x) for x in args.corpus_sizes.split(",")]:
        hours = n / best["windows_per_s"] / 3600
        proj[n] = {"forward_hours": hours, "storage_gb": n * mb / 1e3}
        log(f"{n:>7} windows: forward ~{hours:5.2f} h at bs={best['batch_size']}, "
            f"storage {n*mb/1e3:6.1f} GB at one layer (fp16)")
    log(f"stored tokens per window {real_tokens:.0f} -> {mb:.3f} MB per window per layer")
    log("disk write speed from the shards stage may bound the real wall-clock")
    report["project"] = {"best_batch": best, "mb_per_window": mb,
                         "corpus": proj}


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="local path or hub id")
    ap.add_argument("--revision", default=None,
                    help=f"hub revision; use {PINNED_REVISION} for AgroNT")
    ap.add_argument("--windows", required=True, help="FASTA of windows")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto", help="fp32|bf16|fp16|auto")
    ap.add_argument("--stages", default="all")
    ap.add_argument("--layers", default="4,12,20,30,40")
    ap.add_argument("--shard-layer", type=int, default=None)
    ap.add_argument("--max-seqs", type=int, default=None)
    ap.add_argument("--batch-sizes", default="1,2,4,8,16,32,64,128")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--precision-n", type=int, default=32)
    ap.add_argument("--reduced", default="bf16,fp16")
    ap.add_argument("--extract-batch", type=int, default=16)
    ap.add_argument("--shard-windows", type=int, default=100)
    ap.add_argument("--simulate-kill-after", type=int, default=None,
                    help="exit abruptly while writing shard N (restart test)")
    ap.add_argument("--corpus-sizes", default="80000,150000")
    ap.add_argument("--strict", action="store_true",
                    help="fail if the config differs from AgroNT's")
    ap.add_argument("--outdir", default="bench_out")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if a.device == "auto" else a.device)
    if a.dtype == "auto":
        a.dtype = ("bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported()
                   else "fp32")
    stages = (["smoke", "tokens", "sweep", "precision", "hooks", "determinism",
               "shards", "project"] if a.stages == "all" else a.stages.split(","))
    layers = [int(x) for x in a.layers.split(",")]

    report = {"environment": environment(), "args": vars(a)}
    log(f"device {device}  dtype {a.dtype}  stages {stages}")
    log(json.dumps(report["environment"]))

    ids, seqs = read_fasta(a.windows, a.max_seqs)
    log(f"loaded {len(seqs)} windows from {a.windows}")
    tok, model = load(a.model, a.revision, device, DTYPES[a.dtype])
    n_layers = model.config.num_hidden_layers
    bad = [k for k in layers if not 1 <= k <= n_layers]
    if bad:
        raise SystemExit(f"layers {bad} outside 1..{n_layers}")

    for st in stages:
        log(f"\n=== {st} ===")
        if st == "smoke":
            stage_smoke(a, tok, model, report)
        elif st == "tokens":
            stage_tokens(a, tok, ids, seqs, report)
        elif st == "sweep":
            stage_sweep(a, tok, model, seqs, device, report)
        elif st == "precision":
            stage_precision(a, tok, a.model, a.revision, seqs, layers, device, report)
        elif st == "hooks":
            stage_hooks(a, tok, model, seqs, layers, device, report)
        elif st == "determinism":
            stage_determinism(a, tok, model, seqs, layers, device, report)
        elif st == "shards":
            stage_shards(a, tok, model, ids, seqs,
                         a.shard_layer or layers[len(layers) // 2], device, report)
        elif st == "project":
            stage_project(a, report)
        else:
            raise SystemExit(f"unknown stage {st}")

        path = os.path.join(a.outdir, "benchmark_report.json")
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
    log(f"\nreport written to {os.path.join(a.outdir, 'benchmark_report.json')}")


if __name__ == "__main__":
    main()
