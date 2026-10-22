#!/usr/bin/env python3
"""
Production activation extractor (Task 7, deliverable D5).

Runs the AgroNT base transformer (no masked-LM head) over a FASTA of windows
written by extract_windows.py, captures the output of the chosen block(s)
with forward hooks, and writes every real token's vector to an append-only,
checksummed shard store with a per-window manifest.

Settings fixed by the pipeline:
  compute   float32 forward pass; storage float16. Any value that is not
            finite or has |x| > 65,504 stops the run before it is written.
  tokens    real 6-mer tokens only. CLS, EOS, BOS and padding are dropped.
            Every window must be N-free (ACGT only) and a multiple of 6 bp,
            so it tokenises into exactly length/6 six-mers (167 at 1,002 bp).
  site      layer k = output of encoder.layer[k-1] = hidden_states[k],
            checked against hidden_states at every start. k is 1..39: 0 is
            the embedding output and 40 differs from hidden_states[40]
            because of the final layer norm.

Store layout, one folder per layer:
  <outdir>/L13/run_info.json      settings, input and model fingerprints
  <outdir>/L13/manifest.jsonl     one line per shard, with per-window fields
  <outdir>/L13/shard_00000.npy    float16 [rows, 1500], windows in input order
  <outdir>/L13/windows_index.tsv  one row per window (written on completion)
  <outdir>/L13/COMPLETE.json      totals, written only when nothing is missing

Per-window fields: window_id, gene_id, group_id, region_class, chrom,
subgenome, strand, start, end, tss, length_bp, seq_sha256, n_tokens,
token_bp, shard, row_start, row_end. Token i of a window covers window bases
6i+1..6i+6 counted from the distal end; token_span() maps it to the genome.

Resume and storage safety:
  - shard i always holds windows i*S .. (i+1)*S-1 of the input, so a rerun
    writes exactly the shards that are missing, in every requested layer
  - a shard is written to .tmp, fsynced, hashed, atomically renamed, then
    appended to the manifest; on restart, .tmp files and shards missing from
    the manifest are removed, a half-written last manifest line is dropped,
    and every recorded shard is re-hashed before anything new is written
  - a rerun with different settings, inputs or model files is refused
  - free disk space is checked before starting and before every shard

Usage (lab GPU):
  python scripts/extract_activations.py --model models/agro-nt-1b \
      --windows data/corpus_windows_L1002.fa \
      --manifest data/corpus_promoters_L1002.tsv \
      --exclusion data/exclusion_manifest_L1002.tsv \
      --expect-content-hash 97a889a0eb2aa118 \
      --region-class promoter --layers 13 --outdir activations

  Add --dry-run to check inputs and the storage projection without loading
  the model weights or writing anything.
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark  # noqa: E402
from benchmark import (PINNED_REVISION, encode, encoder_layers,  # noqa: E402
                       environment, real_token_mask, sha256_file)

FP16_MAX = 65504.0
TOKEN_BP = 6
SHARD_RE = re.compile(r"^shard_(\d{5})\.npy$")
HEADER_RE = re.compile(r"^(\S+) (\S+):(\d+)-(\d+)\(([+-])\) tss=(\d+) len=(\d+)")
# run_info entries that must be identical before a store is resumed
RESUME_KEYS = ("layer", "region_class", "length_bp", "shard_windows",
               "compute_dtype", "storage_dtype", "revision", "window_ids_sha256",
               "windows_content_sha256", "model_files_sha256")


def log(msg=""):
    print(msg, flush=True)


# --------------------------------------------------------------------- inputs

def subgenome(chrom):
    """chr1A -> A, chrUn -> Un. Anything else is an error, not a guess."""
    if chrom == "chrUn":
        return "Un"
    m = re.fullmatch(r"chr[1-7]([ABD])", chrom)
    if not m:
        raise SystemExit(f"unrecognised chromosome name {chrom!r}")
    return m.group(1)


def token_span(rec, i):
    """1-based genome coordinates (lo, hi) covered by token i of a window."""
    if rec["strand"] == "+":
        lo = rec["start"] + TOKEN_BP * i
        return lo, lo + TOKEN_BP - 1
    hi = rec["end"] - TOKEN_BP * i
    return hi - TOKEN_BP + 1, hi


def read_windows(path):
    """Windows FASTA from extract_windows.py, headers parsed, order kept."""
    recs, seq, head = [], [], None

    def flush():
        m = HEADER_RE.match(head)
        if not m:
            raise SystemExit(f"unexpected FASTA header {head!r}; "
                             "use windows written by extract_windows.py")
        wid, chrom, s, e, strand, tss, length = m.groups()
        recs.append({"window_id": wid, "chrom": chrom, "start": int(s),
                     "end": int(e), "strand": strand, "tss": int(tss),
                     "length_bp": int(length), "seq": "".join(seq).upper()})

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if head is not None:
                    flush()
                head, seq = line[1:], []
            elif line:
                seq.append(line)
    if head is not None:
        flush()
    return recs


def load_inputs(a):
    recs = read_windows(a.windows)
    if a.max_windows:
        recs = recs[:a.max_windows]
    if not recs:
        raise SystemExit(f"no windows in {a.windows}")
    ids = [r["window_id"] for r in recs]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate window IDs in the windows FASTA")

    m = pd.read_csv(a.manifest, sep="\t", dtype={"chrom": str, "strand": str})
    key = "window_id" if "window_id" in m.columns else "gene_id"
    need = {key, "gene_id", "chrom", "strand", "tss", "group_id"}
    if need - set(m.columns):
        raise SystemExit(f"manifest lacks columns {sorted(need - set(m.columns))}")
    if m[key].duplicated().any():
        raise SystemExit(f"duplicate {key} values in {a.manifest}")
    if "region_class" not in m.columns and not a.region_class:
        raise SystemExit("give --region-class (the manifest has no region_class column)")
    m = m.set_index(key, drop=False)

    problems = []
    for r in recs:
        wid = r["window_id"]
        if wid not in m.index:
            problems.append(f"{wid}: not in manifest")
            continue
        row = m.loc[wid]
        if (row["chrom"], row["strand"], int(row["tss"])) != (r["chrom"], r["strand"], r["tss"]):
            problems.append(f"{wid}: chrom/strand/tss differ between FASTA and manifest")
        r["gene_id"] = row["gene_id"]
        r["group_id"] = row["group_id"]
        r["region_class"] = row["region_class"] if "region_class" in m.columns else a.region_class
        r["subgenome"] = subgenome(r["chrom"])
        seq = r["seq"]
        if len(seq) != r["length_bp"] or len(seq) != a.length:
            problems.append(f"{wid}: length {len(seq)}, expected {a.length}")
        if len(seq) % TOKEN_BP:
            problems.append(f"{wid}: length {len(seq)} is not a multiple of {TOKEN_BP}")
        if len(seq) - sum(seq.count(c) for c in "ACGT"):
            problems.append(f"{wid}: non-ACGT base (N rule)")
        r["seq_sha256"] = hashlib.sha256(seq.encode()).hexdigest()
        r["n_tokens"] = len(seq) // TOKEN_BP
    if problems:
        raise SystemExit(f"{len(problems)} input problems, nothing written:\n  "
                         + "\n  ".join(problems[:20]))

    if a.exclusion:
        ex = set(pd.read_csv(a.exclusion, sep="\t")["gene_id"])
        hit = [r["gene_id"] for r in recs if r["gene_id"] in ex]
        if hit:
            raise SystemExit(f"{len(hit)} windows belong to excluded genes "
                             f"(e.g. {hit[:5]}); nothing written")
    elif not a.no_exclusion_check:
        raise SystemExit("give --exclusion <exclusion manifest>, or --no-exclusion-check "
                         "for windows that are evaluation genes on purpose")

    h = hashlib.sha256()
    for r in recs:
        h.update(r["seq"].encode())
    content = h.hexdigest()
    if a.expect_content_hash and not content.startswith(a.expect_content_hash):
        raise SystemExit(f"windows content hash {content[:16]} does not match "
                         f"--expect-content-hash {a.expect_content_hash}")
    ids_hash = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    return recs, content, ids_hash


# ---------------------------------------------------------------- model setup

def model_fingerprint(model_path):
    """SHA-256 of every file in a local model folder (hub IDs: revision only)."""
    if not os.path.isdir(model_path):
        return {}
    out = {}
    for root, _dirs, files in os.walk(model_path):
        for f in sorted(files):
            p = os.path.join(root, f)
            out[os.path.relpath(p, model_path).replace(os.sep, "/")] = sha256_file(p)
    return dict(sorted(out.items()))


def load_base(model_path, revision, device):
    from transformers import AutoModel, AutoTokenizer
    kw = {"revision": revision} if revision else {}
    tok = AutoTokenizer.from_pretrained(model_path, **kw)
    model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float32,
                                      add_pooling_layer=False, **kw)
    model.eval().to(device)
    return tok, model


def check_config(cfg):
    got = {k: getattr(cfg, k, None) for k in benchmark.EXPECTED}
    bad = {k: (got[k], v) for k, v in benchmark.EXPECTED.items() if got[k] != v}
    if bad:
        raise SystemExit(f"model configuration differs from the pinned AgroNT values: {bad}")
    return got


def check_layers(layers, n_layers):
    bad = [k for k in layers if not 1 <= k <= n_layers - 1]
    if bad:
        raise SystemExit(f"layers {bad} not allowed: use 1..{n_layers - 1} "
                         f"(0 is the embedding output; {n_layers} is captured "
                         "before the final layer norm, so it is not hidden_states[-1])")
    if len(set(layers)) != len(layers):
        raise SystemExit("a layer is listed twice")


def check_ids(input_ids, vocab_size):
    top = int(input_ids.max())
    if top >= vocab_size:
        raise SystemExit(f"token ID {top} is outside the model vocabulary ({vocab_size})")


def check_tokens(input_ids, keep, tok_len, recs):
    """Kept positions must be exactly the window's n_tokens six-mers."""
    counts = keep.sum(dim=1).tolist()
    for j, r in enumerate(recs):
        if counts[j] != r["n_tokens"]:
            raise SystemExit(f"{r['window_id']}: {counts[j]} real tokens, "
                             f"expected {r['n_tokens']}")
    lens = tok_len[input_ids[keep]]
    if bool((lens != TOKEN_BP).any()):
        raise SystemExit("a kept token is not a 6-mer; the window is not cleanly framed")


def check_activations(x, wid, layer):
    """x: float32 tensor for one window. Refuses anything float16 can't hold."""
    if not bool(torch.isfinite(x).all()):
        raise SystemExit(f"{wid} layer {layer}: non-finite activation; nothing written")
    top = float(x.abs().max())
    if top > FP16_MAX:
        raise SystemExit(f"{wid} layer {layer}: |x| = {top:.1f} exceeds the float16 "
                         f"maximum {FP16_MAX:.0f}; nothing written")


class _Stop(Exception):
    pass


def capture(model, blocks, enc, layers, stop_early=True):
    """Outputs of the hooked blocks; the forward pass stops after the last one."""
    got, last, handles = {}, max(layers), []
    for k in layers:
        def hook(_m, _inp, output, k=k):
            got[k] = output[0] if isinstance(output, tuple) else output
            if stop_early and k == last:
                raise _Stop
        handles.append(blocks[k - 1].register_forward_hook(hook))
    try:
        model(**enc)
    except _Stop:
        pass
    finally:
        for h in handles:
            h.remove()
    return got


def verify_hooks(model, blocks, enc, layers):
    hs = model(**enc, output_hidden_states=True).hidden_states
    got = capture(model, blocks, enc, layers)
    for k in layers:
        diff = float((got[k] - hs[k]).abs().max())
        if diff > 1e-5:
            raise SystemExit(f"hook on block {k} differs from hidden_states[{k}] "
                             f"by {diff:.3e}")
        log(f"hook check layer {k}: matches hidden_states[{k}] (max|diff| {diff:.1e})")


# ---------------------------------------------------------------- shard store

def fsync_dir(path):
    if os.name != "nt":  # directories cannot be fsynced on Windows
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def write_atomic(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_line(path, rec):
    with open(path, "a", newline="\n") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=os.path.dirname(os.path.abspath(__file__)),
                              timeout=10).stdout.strip() or None
    except Exception:
        return None


def open_layer_store(ldir, info, expected_chunks):
    """Create or resume one layer's store. Returns {shard index: record}."""
    os.makedirs(ldir, exist_ok=True)
    info_path = os.path.join(ldir, "run_info.json")
    manifest = os.path.join(ldir, "manifest.jsonl")
    if os.path.exists(info_path):
        with open(info_path) as f:
            old = json.load(f)
        diff = [k for k in RESUME_KEYS if old.get(k) != info.get(k)]
        if diff:
            raise SystemExit(f"{ldir} was written with different {diff}; refusing to mix "
                             "settings. Use a new --outdir.")
    else:
        if os.listdir(ldir):
            raise SystemExit(f"{ldir} has files but no run_info.json; refusing to use it")
        write_atomic(info_path, json.dumps(info, indent=2, default=str))

    for f in os.listdir(ldir):
        if f.startswith("shard_") and f.endswith(".tmp"):
            os.remove(os.path.join(ldir, f))
            log(f"{ldir}: removed incomplete shard {f}")

    done = {}
    if os.path.exists(manifest):
        with open(manifest) as f:
            lines = f.read().split("\n")
        while lines and not lines[-1].strip():
            lines.pop()
        recs = []
        for n, line in enumerate(lines):
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                if n != len(lines) - 1:
                    raise SystemExit(f"{manifest} line {n + 1} is corrupt; do not resume")
                log(f"{ldir}: dropped a half-written last manifest line")
                write_atomic(manifest, "".join(json.dumps(r) + "\n" for r in recs))
        for rec in recs:
            path = os.path.join(ldir, rec["file"])
            if not os.path.exists(path) or sha256_file(path) != rec["sha256"]:
                raise SystemExit(f"checksum failure on {path}; the store is corrupt, "
                                 "do not resume")
            i = rec["shard"]
            if i in done or i >= len(expected_chunks) or \
                    [w["window_id"] for w in rec["windows"]] != expected_chunks[i]:
                raise SystemExit(f"{path}: windows differ from this input; do not resume")
            done[i] = rec
        log(f"{ldir}: {len(done)} shards verified by checksum")

    for f in os.listdir(ldir):
        m = SHARD_RE.match(f)
        if m and int(m.group(1)) not in done:
            os.remove(os.path.join(ldir, f))
            log(f"{ldir}: removed shard {f} that never reached the manifest")
    return done


def finish_layer(ldir, n_windows):
    manifest = os.path.join(ldir, "manifest.jsonl")
    with open(manifest) as f:
        recs = sorted((json.loads(line) for line in f if line.strip()),
                      key=lambda r: r["shard"])
    rows = [dict(w, shard=r["file"]) for r in recs for w in r["windows"]]
    missing = n_windows - len(rows)
    if missing:
        log(f"{ldir}: {missing} windows missing; not marked complete")
        return False
    cols = ["window_id", "gene_id", "group_id", "region_class", "chrom", "subgenome",
            "strand", "start", "end", "tss", "length_bp", "seq_sha256", "n_tokens",
            "token_bp", "shard", "row_start", "row_end"]
    write_atomic(os.path.join(ldir, "windows_index.tsv"),
                 pd.DataFrame(rows)[cols].to_csv(sep="\t", index=False, lineterminator="\n"))
    total = {"windows": len(rows), "missing": 0, "shards": len(recs),
             "rows": sum(r["shape"][0] for r in recs),
             "bytes": sum(os.path.getsize(os.path.join(ldir, r["file"])) for r in recs),
             "manifest_sha256": sha256_file(manifest),
             "completed": datetime.datetime.now().isoformat(timespec="seconds")}
    write_atomic(os.path.join(ldir, "COMPLETE.json"), json.dumps(total, indent=2))
    log(f"{ldir}: complete, {total['windows']} windows, {total['rows']} token rows, "
        f"{total['bytes'] / 2**30:.2f} GB, missing 0")
    return True


def free_bytes(path):
    return shutil.disk_usage(path).free


# ----------------------------------------------------------------------- main

@torch.no_grad()
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None,
                    help=f"hub revision; AgroNT is pinned at {PINNED_REVISION}")
    ap.add_argument("--windows", required=True, help="FASTA from extract_windows.py")
    ap.add_argument("--manifest", required=True,
                    help="TSV with gene_id, chrom, strand, tss, group_id")
    ap.add_argument("--exclusion", default=None, help="exclusion manifest TSV")
    ap.add_argument("--no-exclusion-check", action="store_true")
    ap.add_argument("--expect-content-hash", default=None,
                    help="prefix of the content hash printed by extract_windows.py")
    ap.add_argument("--region-class", default=None,
                    help="e.g. promoter; needed when the manifest has no region_class")
    ap.add_argument("--length", type=int, default=1002)
    ap.add_argument("--layers", required=True, help="comma-separated, e.g. 13 or 13,30")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--shard-windows", type=int, default=500)
    ap.add_argument("--reserve-gb", type=float, default=10.0,
                    help="free space that must remain after the projected writes")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-windows", type=int, default=None, help="first N windows only")
    ap.add_argument("--dry-run", action="store_true",
                    help="check inputs and storage only; load no weights, write nothing")
    ap.add_argument("--simulate-kill-after", type=int, default=None,
                    help="exit abruptly while writing shard N (restart test)")
    a = ap.parse_args(argv)

    layers = sorted(int(x) for x in a.layers.split(","))
    recs, content_hash, ids_hash = load_inputs(a)
    n_tok = sum(r["n_tokens"] for r in recs)
    S = a.shard_windows
    chunks = [recs[i:i + S] for i in range(0, len(recs), S)]
    expected_ids = [[r["window_id"] for r in c] for c in chunks]
    log(f"{len(recs)} windows ({n_tok} tokens), {len(chunks)} shards of up to {S}; "
        f"content hash {content_hash[:16]}")

    from transformers import AutoConfig
    kw = {"revision": a.revision} if a.revision else {}
    cfg_got = check_config(AutoConfig.from_pretrained(a.model, **kw))
    check_layers(layers, cfg_got["num_hidden_layers"])
    hidden = cfg_got["hidden_size"]
    row_bytes = hidden * 2

    outdir = os.path.abspath(a.outdir)
    if os.path.isdir(a.model) and (outdir + os.sep).startswith(os.path.abspath(a.model) + os.sep):
        raise SystemExit("--outdir must not be inside the model folder")

    if a.dry_run:
        need = n_tok * row_bytes * len(layers)
        probe = outdir if os.path.isdir(outdir) else os.path.dirname(outdir) or "."
        log(f"dry run: up to {need / 2**30:.2f} GB for layers {layers}; "
            f"{free_bytes(probe) / 2**30:.1f} GB free at {probe}; reserve {a.reserve_gb} GB")
        return 0

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if a.device == "auto" else a.device)
    log("fingerprinting model files ...")
    fingerprint = model_fingerprint(a.model)
    tok, model = load_base(a.model, a.revision, device)
    check_config(model.config)
    blocks, block_path = encoder_layers(model)
    vocab = model.config.vocab_size
    tok_len = torch.zeros(len(tok), dtype=torch.long)
    for t, i in tok.get_vocab().items():
        tok_len[i] = len(t)
    tok_len = tok_len.to(device)

    os.makedirs(outdir, exist_ok=True)
    base_info = {
        "script": os.path.basename(__file__), "script_sha256": sha256_file(__file__),
        "git_commit": git_commit(),
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "region_class": a.region_class, "length_bp": a.length, "shard_windows": S,
        "compute_dtype": "float32", "storage_dtype": "float16",
        "tokens_stored": "real 6-mer tokens only; CLS, EOS, BOS and padding dropped",
        "token_bp": TOKEN_BP, "model": a.model, "revision": a.revision,
        "model_files_sha256": fingerprint, "model_config": cfg_got,
        "model_class": type(model).__name__, "tokenizer_len": len(tok),
        "windows_file": a.windows, "windows_file_sha256": sha256_file(a.windows),
        "windows_content_sha256": content_hash, "window_ids_sha256": ids_hash,
        "n_windows": len(recs), "n_tokens": n_tok,
        "manifest_file": a.manifest, "manifest_sha256": sha256_file(a.manifest),
        "exclusion_file": a.exclusion,
        "exclusion_sha256": sha256_file(a.exclusion) if a.exclusion else None,
        "environment": environment(),
    }
    stores = {}
    for k in layers:
        info = dict(base_info, layer=k,
                    hook_site=f"output of {block_path}[{k - 1}] = hidden_states[{k}]")
        ldir = os.path.join(outdir, f"L{k}")
        stores[k] = (ldir, open_layer_store(ldir, info, expected_ids))

    todo = [i for i in range(len(chunks)) if any(i not in done for _, done in stores.values())]
    need = sum(sum(r["n_tokens"] for r in chunks[i]) * row_bytes
               for i in todo for _, done in stores.values() if i not in done)
    free = free_bytes(outdir)
    log(f"to write: {len(todo)} shards, {need / 2**30:.2f} GB; free {free / 2**30:.1f} GB; "
        f"reserve {a.reserve_gb} GB")
    if free - need < a.reserve_gb * 2**30:
        raise SystemExit("not enough free space for the projected writes plus the reserve; "
                         "nothing new written")

    if todo:
        first = chunks[todo[0]][:min(4, len(chunks[todo[0]]))]
        verify_hooks(model, blocks, encode(tok, [r["seq"] for r in first], device), layers)

    t0, wrote = time.perf_counter(), 0
    for n, i in enumerate(todo):
        chunk = chunks[i]
        targets = [k for k in layers if i not in stores[k][1]]
        shard_bytes = sum(r["n_tokens"] for r in chunk) * row_bytes * len(targets)
        if free_bytes(outdir) - shard_bytes < a.reserve_gb * 2**30:
            raise SystemExit(f"free space fell below the {a.reserve_gb} GB reserve before "
                             f"shard {i}; free space, then rerun to resume")
        acts = {k: [] for k in targets}
        for b in range(0, len(chunk), a.batch):
            part = chunk[b:b + a.batch]
            enc = encode(tok, [r["seq"] for r in part], device)
            check_ids(enc["input_ids"], vocab)
            keep = real_token_mask(tok, enc)
            check_tokens(enc["input_ids"], keep, tok_len, part)
            got = capture(model, blocks, enc, targets)
            for k in targets:
                for j, r in enumerate(part):
                    x = got[k][j][keep[j]].float()
                    check_activations(x, r["window_id"], k)
                    acts[k].append(x.cpu().numpy().astype(np.float16))

        windows, row = [], 0
        for r in chunk:
            windows.append({f: r[f] for f in (
                "window_id", "gene_id", "group_id", "region_class", "chrom", "subgenome",
                "strand", "start", "end", "tss", "length_bp", "seq_sha256", "n_tokens")}
                | {"token_bp": TOKEN_BP, "row_start": row, "row_end": row + r["n_tokens"]})
            row += r["n_tokens"]
        for k in targets:
            ldir = stores[k][0]
            arr = np.concatenate(acts[k], axis=0)
            name = f"shard_{i:05d}.npy"
            tmp = os.path.join(ldir, name + ".tmp")
            with open(tmp, "wb") as f:
                np.save(f, arr)
                if a.simulate_kill_after is not None and i >= a.simulate_kill_after:
                    f.flush()
                    log(f"simulated kill while writing {ldir}/{name}")
                    os._exit(9)
                f.flush()
                os.fsync(f.fileno())
            digest = sha256_file(tmp)
            os.replace(tmp, os.path.join(ldir, name))
            fsync_dir(ldir)
            rec = {"shard": i, "file": name, "sha256": digest, "shape": list(arr.shape),
                   "dtype": "float16", "layer": k, "windows": windows}
            append_line(os.path.join(ldir, "manifest.jsonl"), rec)
            stores[k][1][i] = rec
            wrote += arr.nbytes
        dt = time.perf_counter() - t0
        log(f"shard {i} ({n + 1}/{len(todo)}) layers {targets}; "
            f"{wrote / 2**30:.2f} GB in {dt:.0f}s; "
            f"eta {dt / (n + 1) * (len(todo) - n - 1) / 60:.1f} min")

    ok = all([finish_layer(stores[k][0], len(recs)) for k in layers])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
