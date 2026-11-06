#!/usr/bin/env python3
"""
Label-blind SAE pilot (proposal sec. 6.4, deliverable D6): prepare the data,
train ReLU/L1 and TopK sparse autoencoders on stored layer-13 token vectors.

Reads only activation stores and their windows_index.tsv. It never reads SD6,
splits or drought files; no corpus window belongs to an evaluation gene.

  prepare   held-out assignment (10% of windows, whole homologue groups kept
            together), normalisation constants from training windows, the
            held-out token array and a fixed monitoring sample
  train     one run; checkpoints with optimiser state on a wall-clock interval
            and resumes from the exact batch
  grid      every run of a JSON grid, one at a time, skipping finished runs

Stop and restart: rerun the same command after any stop and it continues.
  - prepare writes prep.json last and skips itself once that file exists
  - a run saves weights, optimiser and position every --ckpt-minutes, atomically;
    a hard kill loses at most that much work, and the resumed run takes the same
    batches from the same position
  - to stop on purpose, create <outdir>/STOP (or press Ctrl+C): the run saves
    and exits. Delete STOP before restarting

Layout under --outdir:
  prep.json, norm.npz, heldout_windows.tsv, heldout_tokens.npy,
  heldout_tokens_class.npy, monitor_tokens.npy, shard_index.json
  runs/<name>/config.json, resume.pt, log.tsv, final.pt, DONE.json

Usage:
  python scripts/train_sae.py prepare --stores activations/L13 activations/regions/L13 --outdir sae_pilot
  python scripts/train_sae.py train --outdir sae_pilot --name relu_l1-2_s1 --kind relu --l1 2 --seed 1
  python scripts/train_sae.py grid --outdir sae_pilot --grid docs/sae_pilot_grid.json
"""
import argparse
import datetime
import hashlib
import json
import os
import platform
import subprocess
import threading
import time
from queue import Queue

import numpy as np
import pandas as pd
import torch

HELDOUT_SEED = 20261105
HELDOUT_FRAC = 0.10
MONITOR_TOKENS = 65536
INPUT_DIM = 1500


def log(msg):
    print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_atomic_torch(path, obj):
    """A stop at any moment leaves either the previous file or the complete new one."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return None


# ------------------------------------------------------------------- prepare

def assign_heldout(windows, frac=HELDOUT_FRAC, seed=HELDOUT_SEED):
    """Boolean per window: held out. Whole group_ids go together, across classes."""
    groups = np.sort(windows["group_id"].unique())
    rng = np.random.default_rng(seed)
    n = int(round(frac * len(groups)))
    out = set(groups[rng.permutation(len(groups))[:n]])
    return windows["group_id"].isin(out).to_numpy()


def ranges(rows):
    """Merge [(a, b), ...] sorted by a into contiguous ranges."""
    out = []
    for a, b in rows:
        if out and out[-1][1] == a:
            out[-1][1] = b
        else:
            out.append([a, b])
    return out


def load_rows(path, rs):
    """float16 rows of one shard for the given ranges, as a new array."""
    a = np.load(path, mmap_mode="r")
    return np.concatenate([np.asarray(a[x:y]) for x, y in rs]) if rs else np.empty((0, a.shape[1]), a.dtype)


def cmd_prepare(a):
    os.makedirs(a.outdir, exist_ok=True)
    prep_path = os.path.join(a.outdir, "prep.json")
    tables, stores = [], {}
    for s in a.stores:
        done = os.path.join(s, "COMPLETE.json")
        if not os.path.exists(done):
            raise SystemExit(f"{s}: no COMPLETE.json; the store is not finished")
        info = load_json(done)
        w = pd.read_csv(os.path.join(s, "windows_index.tsv"), sep="\t", dtype={"chrom": str})
        if len(w) != info["windows"] or (w["row_end"] - w["row_start"]).sum() != info["rows"]:
            raise SystemExit(f"{s}: windows_index.tsv disagrees with COMPLETE.json")
        w["store"] = s
        tables.append(w)
        stores[s] = {"windows": info["windows"], "rows": info["rows"],
                     "manifest_sha256": sha256_file(os.path.join(s, "manifest.jsonl"))}
    if os.path.exists(prep_path):
        if load_json(prep_path)["stores"] != stores:
            raise SystemExit(f"{prep_path} was prepared from different stores; refusing to continue")
        log("prepare: already complete")
        return
    w = pd.concat(tables, ignore_index=True)
    if w["window_id"].duplicated().any():
        raise SystemExit("window ids repeat across stores")
    w["heldout"] = assign_heldout(w)
    log(f"windows {len(w)}, held out {int(w['heldout'].sum())}")
    print(w.groupby("region_class")["heldout"].agg(["size", "mean"]).round(4).to_string(), flush=True)

    classes = sorted(w["region_class"].unique())
    n_held = int((w.loc[w["heldout"], "row_end"] - w.loc[w["heldout"], "row_start"]).sum())
    held = np.lib.format.open_memmap(os.path.join(a.outdir, "heldout_tokens.npy"), mode="w+",
                                     dtype=np.float16, shape=(n_held, INPUT_DIM))
    held_class = np.zeros(n_held, np.int8)
    index, pos = [], 0
    n, sx, sxx = 0, np.zeros(INPUT_DIM), np.zeros(INPUT_DIM)
    for (store, shard), d in w.groupby(["store", "shard"], sort=False):
        path = os.path.join(store, shard)
        d = d.sort_values("row_start")
        tr = ranges([(r.row_start, r.row_end) for r in d[~d["heldout"]].itertuples()])
        hd = d[d["heldout"]]
        arr = np.load(path)
        for r in hd.itertuples():
            k = r.row_end - r.row_start
            held[pos:pos + k] = arr[r.row_start:r.row_end]
            held_class[pos:pos + k] = classes.index(r.region_class)
            pos += k
        x = np.concatenate([arr[p:q] for p, q in tr]).astype(np.float64) if tr else np.empty((0, INPUT_DIM))
        n += len(x)
        sx += x.sum(0)
        sxx += (x ** 2).sum(0)
        index.append({"path": path, "train": tr, "rows": int(sum(q - p for p, q in tr))})
    held.flush()
    if pos != n_held:
        raise SystemExit("held-out token count mismatch")
    np.save(os.path.join(a.outdir, "heldout_tokens_class.npy"), held_class)

    mean = sx / n
    var_total = float((sxx / n - mean ** 2).sum())   # E||x - mean||^2
    scale = float(np.sqrt(INPUT_DIM / var_total))
    np.savez(os.path.join(a.outdir, "norm.npz"), mean=mean.astype(np.float32), scale=np.float32(scale))
    w.loc[w["heldout"], ["window_id", "group_id", "region_class", "store", "shard", "row_start", "row_end"]].to_csv(
        os.path.join(a.outdir, "heldout_windows.tsv"), sep="\t", index=False)

    idx = np.linspace(0, n_held - 1, MONITOR_TOKENS).astype(np.int64)
    np.save(os.path.join(a.outdir, "monitor_tokens.npy"), np.asarray(held[idx]))
    write_atomic_json(os.path.join(a.outdir, "shard_index.json"), index)
    write_atomic_json(prep_path, {
        "script_sha256": sha256_file(os.path.abspath(__file__)), "git_commit": git_commit(),
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "stores": stores, "heldout_seed": HELDOUT_SEED, "heldout_frac": HELDOUT_FRAC,
        "train_tokens": int(n), "heldout_tokens": n_held, "classes": classes,
        "n_windows": int(len(w)), "n_heldout_windows": int(w["heldout"].sum()),
        "heldout_by_class": {k: int(v) for k, v in w[w["heldout"]]["region_class"].value_counts().items()},
        "mean_norm": float(np.linalg.norm(mean)), "centred_rms_norm": float(np.sqrt(var_total)), "scale": scale})
    log(f"train tokens {n}, held-out tokens {n_held}; mean norm {np.linalg.norm(mean):.1f}, "
        f"centred RMS norm {np.sqrt(var_total):.1f}, scale {scale:.4f}")


# --------------------------------------------------------------------- model

class SAE(torch.nn.Module):
    """x_hat = W_dec f + b_dec, on centred, rescaled input x."""

    def __init__(self, d, m, kind, k=None, generator=None, init_norm=1.0):
        super().__init__()
        if kind not in ("relu", "topk"):
            raise ValueError(kind)
        if kind == "topk" and not k:
            raise ValueError("topk needs k")
        self.kind, self.k = kind, k
        w = torch.randn(m, d, generator=generator)
        w = init_norm * w / w.norm(dim=1, keepdim=True)
        self.W_dec = torch.nn.Parameter(w.clone())       # m x d, each row a feature direction
        self.W_enc = torch.nn.Parameter(w.t().clone())   # d x m, tied at the start only
        self.b_enc = torch.nn.Parameter(torch.zeros(m))
        self.b_dec = torch.nn.Parameter(torch.zeros(d))

    def encode(self, x):
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        if self.kind == "relu":
            return torch.relu(pre)
        v, i = pre.topk(self.k, dim=-1)
        return torch.zeros_like(pre).scatter_(-1, i, torch.relu(v))

    def decode(self, f):
        return f @ self.W_dec + self.b_dec

    def forward(self, x):
        f = self.encode(x)
        return self.decode(f), f

    @torch.no_grad()
    def unit_norm_decoder(self):
        self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8))


def sae_loss(model, x, l1):
    xh, f = model(x)
    rec = ((xh - x) ** 2).sum(-1).mean()
    if model.kind == "relu":
        pen = (f * model.W_dec.norm(dim=1)).sum(-1).mean()
        return rec + l1 * pen, rec, f
    return rec, rec, f


def l1_at(step, total, l1, warm):
    """L1 coefficient, ramped linearly from 0 over the first warm fraction of steps."""
    return l1 if warm <= 0 else l1 * min(1.0, (step + 1) / (warm * total))


def lr_at(step, total, base, warm=0.01, decay_from=0.8):
    if step < warm * total:
        return base * (step + 1) / max(1, warm * total)
    if step >= decay_from * total:
        return base * max(0.0, (total - step) / max(1, total - decay_from * total))
    return base


# --------------------------------------------------------------------- batches

def chunk_plan(n_shards, seed, pass_i, chunk_shards):
    """Shard order of one pass, split into mixing chunks. Deterministic."""
    perm = np.random.default_rng([seed, pass_i]).permutation(n_shards)
    return [perm[i:i + chunk_shards] for i in range(0, n_shards, chunk_shards)]


def load_chunk(index, ids):
    return np.concatenate([load_rows(index[i]["path"], index[i]["train"]) for i in ids])


def batch_stream(index, seed, batch, chunk_shards, start=(0, 0, 0), device="cpu", prefetch=1):
    """Yield (pass, chunk, batch_i, tensor). Fully determined by the seed, so a
    resume at (pass, chunk, batch_i) continues the same sequence."""
    q = Queue(maxsize=prefetch)
    stop = threading.Event()

    def producer():
        p, c0 = start[0], start[1]
        while not stop.is_set():
            plan = chunk_plan(len(index), seed, p, chunk_shards)
            for c in range(c0, len(plan)):
                if stop.is_set():
                    return
                q.put((p, c, load_chunk(index, plan[c])))
            p, c0 = p + 1, 0

    t = threading.Thread(target=producer, daemon=True)
    t.start()
    try:
        first = True
        while True:
            p, c, arr = q.get()
            perm = np.random.default_rng([seed, p, c, 7]).permutation(len(arr))
            n_rows = len(arr)
            x = torch.from_numpy(arr).to(device)
            del arr
            b0 = start[2] if first else 0
            first = False
            for b in range(b0, n_rows // batch):
                yield p, c, b, x[torch.from_numpy(perm[b * batch:(b + 1) * batch]).to(device)].float()
    finally:
        stop.set()
        while not q.empty():
            q.get_nowait()


# ------------------------------------------------------------------- training

def apply_norm(x, mean, scale):
    return (x - mean) * scale


@torch.no_grad()
def monitor(model, x, batch=4096):
    """FVU, L0 and never-fired count on a fixed normalised held-out sample."""
    err, tot, l0 = 0.0, 0.0, 0.0
    fired = torch.zeros(model.W_dec.shape[0], dtype=torch.bool, device=x.device)
    for i in range(0, len(x), batch):
        xb = x[i:i + batch]
        xh, f = model(xb)
        err += ((xh - xb) ** 2).sum().item()
        tot += (xb ** 2).sum().item()
        on = f > 0
        l0 += on.sum().item()
        fired |= on.any(0)
    return err / tot, l0 / len(x), int((~fired).sum())


def train_run(a):
    rdir = os.path.join(a.outdir, "runs", a.name)
    os.makedirs(rdir, exist_ok=True)
    done_path = os.path.join(rdir, "DONE.json")
    if os.path.exists(done_path):
        log(f"{a.name}: already finished")
        return load_json(done_path)

    device = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = a.matmul == "tf32"
    prep = load_json(os.path.join(a.outdir, "prep.json"))
    index = load_json(os.path.join(a.outdir, "shard_index.json"))
    nz = np.load(os.path.join(a.outdir, "norm.npz"))
    mean = torch.from_numpy(nz["mean"]).to(device)
    scale = float(nz["scale"])
    total = a.steps if a.steps else int(a.passes * prep["train_tokens"]) // a.batch

    cfg = dict(kind=a.kind, l1=a.l1 if a.kind == "relu" else None, k=a.k if a.kind == "topk" else None,
               seed=a.seed, width=a.width, batch=a.batch, tokens=total * a.batch, steps=total, lr=a.lr,
               warmup=0.01, decay_from=0.8, grad_clip=a.grad_clip, l1_warmup=a.l1_warmup, init_norm=a.init_norm, chunk_shards=a.chunk_shards,
               matmul=a.matmul, optimizer="Adam", resampling=None, norm_scale=scale,
               stores=prep["stores"], heldout_seed=prep["heldout_seed"])
    cfg_path = os.path.join(rdir, "config.json")
    if os.path.exists(cfg_path):
        old = load_json(cfg_path)
        if {k: old.get(k) for k in cfg} != cfg:
            raise SystemExit(f"{a.name}: settings differ from the existing run; refusing to resume")
    else:
        write_atomic_json(cfg_path, dict(cfg, script_sha256=sha256_file(os.path.abspath(__file__)),
                                         git_commit=git_commit(), python=platform.python_version(),
                                         torch=torch.__version__,
                                         gpu=torch.cuda.get_device_name(0) if device == "cuda" else None))

    torch.manual_seed(a.seed)
    gen = torch.Generator().manual_seed(a.seed)
    model = SAE(INPUT_DIM, a.width, a.kind, a.k, generator=gen,
                init_norm=a.init_norm if a.kind == "relu" else 1.0).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, betas=(0.9, 0.999), fused=device == "cuda")

    step, pos, elapsed_before = 0, (0, 0, 0), 0.0
    resume = os.path.join(rdir, "resume.pt")
    if os.path.exists(resume):
        st = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        step, pos, elapsed_before = st["step"], tuple(st["pos"]), st["elapsed"]
        log(f"{a.name}: resumed at step {step}, position {pos}")

    mon = apply_norm(torch.from_numpy(np.load(os.path.join(a.outdir, "monitor_tokens.npy"))).to(device).float(),
                     mean, scale)
    log_path = os.path.join(rdir, "log.tsv")
    trim_log(log_path, step)

    t0, last_ckpt, last, start_step = time.time(), time.time(), pos, step
    stop_path = os.path.join(a.outdir, "STOP")

    def checkpoint():
        save_resume(resume, model, opt, step, last, elapsed_before + time.time() - t0)

    stream = batch_stream(index, a.seed, a.batch, a.chunk_shards, start=pos, device=device)
    try:
        for p, c, b, xb in stream:
            if step >= total:
                break
            if os.path.exists(stop_path):
                checkpoint()
                raise SystemExit(f"{a.name}: stopped by {stop_path} at step {step}; "
                                 "delete it and rerun the same command to continue")
            x = apply_norm(xb, mean, scale)
            lr = lr_at(step, total, a.lr)
            for g in opt.param_groups:
                g["lr"] = lr
            loss, rec, f = sae_loss(model, x, l1_at(step, total, a.l1, a.l1_warmup))
            if not torch.isfinite(loss):
                raise SystemExit(f"{a.name}: non-finite loss at step {step}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
            opt.step()
            if a.kind == "topk":
                model.unit_norm_decoder()
            step += 1
            last = (p, c, b + 1)

            if step % a.log_every == 0 or step == total:
                fvu, l0, dead = monitor(model, mon)
                with open(log_path, "a") as fh:
                    fh.write(f"{step}\t{p}\t{lr:.3g}\t{loss.item():.4f}\t{rec.item():.4f}\t"
                             f"{(f > 0).float().sum(-1).mean().item():.2f}\t{fvu:.5f}\t{l0:.2f}\t{dead}\t"
                             f"{elapsed_before + time.time() - t0:.0f}\n")
                log(f"{a.name} step {step}/{total} pass {p} mon_fvu {fvu:.4f} L0 {l0:.1f} never_fired {dead} "
                    f"({(time.time() - t0) / max(1, step - start_step) * 1000:.0f} ms/step)")

            if a.kill_after and step == a.kill_after:
                checkpoint()
                raise SystemExit(f"{a.name}: simulated kill after step {step}")
            if time.time() - last_ckpt > a.ckpt_minutes * 60:
                checkpoint()
                last_ckpt = time.time()
            if step >= total:
                break
    except KeyboardInterrupt:
        checkpoint()
        raise SystemExit(f"{a.name}: interrupted at step {step}; rerun the same command to continue")
    finally:
        stream.close()

    save_atomic_torch(os.path.join(rdir, "final.pt"),
                      {"state": {k: v.cpu() for k, v in model.state_dict().items()}, "config": cfg,
                       "mean": mean.cpu(), "scale": scale})
    fvu, l0, dead = monitor(model, mon)
    out = {"name": a.name, "steps": step, "mon_fvu": fvu, "mon_l0": l0, "mon_never_fired": dead,
           "seconds": elapsed_before + time.time() - t0, "finished": datetime.datetime.now().isoformat(timespec="seconds"),
           "passes": step * a.batch / prep["train_tokens"]}
    write_atomic_json(done_path, out)
    if os.path.exists(resume):
        os.remove(resume)
    log(f"{a.name}: done {out}")
    return out


def save_resume(path, model, opt, step, pos, elapsed):
    save_atomic_torch(path, {"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                             "pos": list(pos), "elapsed": elapsed})


LOG_HEADER = "step\tpass\tlr\ttrain_loss\ttrain_rec\ttrain_l0\tmon_fvu\tmon_l0\tmon_never_fired\telapsed_s\n"


def trim_log(path, step):
    """Keep only rows at or before the resumed step, so replayed steps aren't logged twice."""
    rows = []
    if step and os.path.exists(path):
        with open(path) as f:
            rows = [r for r in f.readlines()[1:] if r.endswith("\n") and int(r.split("\t", 1)[0]) <= step]
    with open(path + ".tmp", "w") as f:
        f.write(LOG_HEADER)
        f.writelines(rows)
    os.replace(path + ".tmp", path)


def cmd_grid(a):
    grid = load_json(a.grid)
    for run in grid["runs"]:
        args = argparse.Namespace(**{"l1": 0.0, "k": 0, **vars(a), **grid.get("defaults", {}), **run})
        train_run(args)


def add_train_args(p):
    p.add_argument("--outdir", required=True)
    p.add_argument("--width", type=int, default=6000)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--passes", type=float, default=5.0)
    p.add_argument("--steps", type=int, default=0, help="overrides --passes")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--l1-warmup", type=float, default=0.0, help="fraction of steps to ramp the L1 coefficient")
    p.add_argument("--init-norm", type=float, default=1.0, help="ReLU only: starting length of decoder rows")
    p.add_argument("--chunk-shards", type=int, default=8)
    p.add_argument("--matmul", choices=["fp32", "tf32"], default="tf32")
    p.add_argument("--device", default="auto")
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--ckpt-minutes", type=float, default=2)
    p.add_argument("--kill-after", type=int, default=0, help="test only: exit after this step")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--stores", nargs="+", required=True)
    p.add_argument("--outdir", required=True)
    p = sub.add_parser("train")
    add_train_args(p)
    p.add_argument("--name", required=True)
    p.add_argument("--kind", choices=["relu", "topk"], required=True)
    p.add_argument("--l1", type=float, default=0.0)
    p.add_argument("--k", type=int, default=0)
    p.add_argument("--seed", type=int, required=True)
    p = sub.add_parser("grid")
    add_train_args(p)
    p.add_argument("--grid", required=True)
    a = ap.parse_args()
    if a.cmd == "prepare":
        cmd_prepare(a)
    elif a.cmd == "train":
        train_run(a)
    else:
        cmd_grid(a)


if __name__ == "__main__":
    main()
