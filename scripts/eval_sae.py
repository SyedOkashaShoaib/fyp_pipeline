#!/usr/bin/env python3
"""
Label-blind measurements for the SAE pilot (proposal sec. 6.4). Everything is
computed on held-out corpus windows prepared by train_sae.py; no label is read.

  heldout     per run: FVU overall and by region class, L0, inactive and
              always-on rates, the feature-frequency distribution
  crossseed   per setting: each feature's best decoder cosine in another seed
  loss        masked-LM loss recovered when x_hat replaces layer 13 (15% of real
              tokens masked, the same masks for every condition)
              --unmasked-only is a diagnostic: it swaps layer 13 only at tokens that
              are not masked, because the SAE never trained on <mask> vectors. It
              writes loss_recovered_unmasked.json and is not the gate measurement
  summary     one table of every run and measurement

Outputs: runs/<name>/eval.json, runs/<name>/feature_freq.npy, crossseed.json,
loss_recovered.json, summary.tsv, all under --outdir.

Usage:
  python scripts/eval_sae.py heldout --outdir sae_pilot
  python scripts/eval_sae.py crossseed --outdir sae_pilot
  python scripts/eval_sae.py loss --outdir sae_pilot --model models/agro-nt-1b \
      --windows data/corpus_windows_L1002.fa data/corpus_regions/corpus_regions_L1002.fa \
      --runs relu_x_s1 relu_x_s2 relu_x_s3
  python scripts/eval_sae.py summary --outdir sae_pilot
"""
import argparse
import glob
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_sae as ts  # noqa: E402

LOSS_LAYER = 13
MASK_FRAC = 0.15
LOSS_SEED = 20261106
MATCH_COS = 0.70


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_sae(rdir, device):
    st = torch.load(os.path.join(rdir, "final.pt"), map_location=device, weights_only=False)
    c = st["config"]
    m = ts.SAE(st["state"]["W_dec"].shape[1], c["width"], c["kind"], c["k"]).to(device)
    m.load_state_dict(st["state"])
    return m.eval(), st["mean"].to(device), float(st["scale"]), c


def finished_runs(outdir):
    return sorted(os.path.dirname(p) for p in glob.glob(os.path.join(outdir, "runs", "*", "DONE.json"))
                  if not os.path.basename(os.path.dirname(p)).startswith("smoke_"))


@torch.no_grad()
def heldout_metrics(model, tokens, labels, n_labels, mean, scale, device, batch=8192):
    """tokens: raw float16 array (may be a memmap); labels: region class per token."""
    m = model.W_dec.shape[0]
    fires = torch.zeros(m, dtype=torch.int64, device=device)
    err = np.zeros(n_labels)
    tot = np.zeros(n_labels)
    n = np.zeros(n_labels, dtype=np.int64)
    l0 = 0
    for i in range(0, len(tokens), batch):
        x = (torch.from_numpy(np.asarray(tokens[i:i + batch])).to(device).float() - mean) * scale
        xh, f = model(x)
        on = f > 0
        fires += on.sum(0)
        l0 += int(on.sum())
        e = ((xh - x) ** 2).sum(1).cpu().numpy()
        t = (x ** 2).sum(1).cpu().numpy()
        lab = labels[i:i + batch]
        np.add.at(err, lab, e)
        np.add.at(tot, lab, t)
        np.add.at(n, lab, 1)
    freq = fires.cpu().numpy() / len(tokens)
    return dict(fvu=float(err.sum() / tot.sum()), fvu_by_class=(err / np.maximum(tot, 1e-12)).tolist(),
                tokens_by_class=n.tolist(), l0=l0 / len(tokens), inactive_frac=float((freq == 0).mean()),
                always_on_frac=float((freq > 0.5).mean()), n_tokens=int(len(tokens))), freq


def cmd_heldout(a):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prep = load_json(os.path.join(a.outdir, "prep.json"))
    classes = prep["classes"]
    tokens = np.load(os.path.join(a.outdir, "heldout_tokens.npy"), mmap_mode="r")
    labels = np.load(os.path.join(a.outdir, "heldout_tokens_class.npy"))
    edges = np.r_[0, np.logspace(-7, 0, 15)]
    for rdir in finished_runs(a.outdir):
        out = os.path.join(rdir, "eval.json")
        if os.path.exists(out) and not a.force:
            continue
        model, mean, scale, cfg = load_sae(rdir, device)
        r, freq = heldout_metrics(model, tokens, labels, len(classes), mean, scale, device)
        r["fvu_by_class"] = dict(zip(classes, r["fvu_by_class"]))
        r["tokens_by_class"] = dict(zip(classes, r["tokens_by_class"]))
        r["freq_hist_edges"] = edges.tolist()
        r["freq_hist"] = np.histogram(freq[freq > 0], bins=edges)[0].tolist()
        np.save(os.path.join(rdir, "feature_freq.npy"), freq)
        ts.write_atomic_json(out, r)
        ts.log(f"{os.path.basename(rdir)}: FVU {r['fvu']:.4f} L0 {r['l0']:.1f} inactive {r['inactive_frac']:.3f} "
               f"always-on {r['always_on_frac']:.4f}")


@torch.no_grad()
def best_cos(a, b, chunk=512):
    """For each row of a, the highest cosine with any row of b."""
    a = a / a.norm(dim=1, keepdim=True).clamp_min(1e-8)
    b = b / b.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return torch.cat([(a[i:i + chunk] @ b.T).max(1).values for i in range(0, len(a), chunk)])


def cmd_crossseed(a):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    groups = {}
    for rdir in finished_runs(a.outdir):
        c = load_json(os.path.join(rdir, "config.json"))
        groups.setdefault((c["kind"], c["l1"], c["k"], c["steps"]), []).append(rdir)
    results = []
    for (kind, l1, k, steps), dirs in sorted(groups.items(), key=str):
        if len(dirs) < 2:
            continue
        cos_all, cos_active, matched = [], [], []
        for d1, d2 in itertools.permutations(dirs, 2):
            m1, *_ = load_sae(d1, device)
            m2, *_ = load_sae(d2, device)
            c12 = best_cos(m1.W_dec.data, m2.W_dec.data).cpu().numpy()
            cos_all.append(float(c12.mean()))
            matched.append(float((c12 >= MATCH_COS).mean()))
            fp = os.path.join(d1, "feature_freq.npy")
            if os.path.exists(fp):
                cos_active.append(float(c12[np.load(fp) > 0].mean()))
        r = dict(kind=kind, l1=l1, k=k, steps=steps, runs=[os.path.basename(d) for d in dirs],
                 mean_max_cos=float(np.mean(cos_all)), mean_max_cos_active=float(np.mean(cos_active)) if cos_active else None,
                 matched_frac=float(np.mean(matched)), pairs=len(cos_all))
        results.append(r)
        ts.log(str(r))
    ts.write_atomic_json(os.path.join(a.outdir, "crossseed.json"), results)


# ------------------------------------------------------- masked-LM loss recovered

def read_fasta_subset(paths, ids):
    want, out, cur, buf = set(ids), {}, None, []
    for p in paths:
        with open(p) as f:
            for line in f:
                if line.startswith(">"):
                    if cur:
                        out[cur] = "".join(buf)
                    wid = line[1:].split()[0]
                    cur, buf = (wid if wid in want else None), []
                elif cur:
                    buf.append(line.strip())
        if cur:
            out[cur] = "".join(buf)
        cur, buf = None, []
    missing = want - set(out)
    if missing:
        raise SystemExit(f"{len(missing)} windows not found in the FASTA files, e.g. {sorted(missing)[:3]}")
    return out


def make_masks(keep, frac, seed):
    """Boolean [windows, positions]: which real tokens are masked, the same every call."""
    rng = np.random.default_rng(seed)
    out = np.zeros(keep.shape, bool)
    for i in range(keep.shape[0]):
        real = np.flatnonzero(keep[i])
        out[i, rng.choice(real, size=int(round(frac * len(real))), replace=False)] = True
    return out


def layer_output_tensor(output):
    return output[0] if isinstance(output, tuple) else output


def replace_output(output, new):
    return (new,) + tuple(output[1:]) if isinstance(output, tuple) else new


@torch.no_grad()
def masked_loss(model, block, batches, transform=None):
    """Mean cross-entropy over masked tokens. transform maps real-token vectors
    to replacement vectors in raw layer units; None leaves the layer as it is."""
    handle = None
    state = {}
    if transform is not None:
        def hook(_m, _i, output):
            h = layer_output_tensor(output).clone()
            k = state["keep"]
            h[k] = transform(h[k])
            return replace_output(output, h)
        handle = block.register_forward_hook(hook)
    total, count = 0.0, 0
    try:
        for b in batches:
            state["keep"] = b["keep"]
            logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
            sel = b["masked"]
            total += torch.nn.functional.cross_entropy(logits[sel], b["labels"][sel], reduction="sum").item()
            count += int(sel.sum())
    finally:
        if handle:
            handle.remove()
    return total / count


def loss_recovered(l_abl, l_sae, l_orig):
    return (l_abl - l_sae) / (l_abl - l_orig)


def cmd_loss(a):
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    import benchmark
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False
    held = pd.read_csv(os.path.join(a.outdir, "heldout_windows.tsv"), sep="\t")
    rng = np.random.default_rng(LOSS_SEED)
    pick = held.iloc[np.sort(rng.choice(len(held), size=min(a.n_windows, len(held)), replace=False))]
    seqs_by_id = read_fasta_subset(a.windows, pick["window_id"])
    seqs = [seqs_by_id[w] for w in pick["window_id"]]

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForMaskedLM.from_pretrained(a.model, torch_dtype=torch.float32).eval().to(device)
    blocks, _ = benchmark.encoder_layers(model)
    block = blocks[LOSS_LAYER - 1]

    enc = benchmark.encode(tok, seqs, "cpu")
    keep = benchmark.real_token_mask(tok, enc)
    masked = torch.from_numpy(make_masks(keep.numpy(), MASK_FRAC, LOSS_SEED))
    labels = enc["input_ids"].clone()
    ids = enc["input_ids"].clone()
    ids[masked] = tok.mask_token_id
    swap = keep & ~masked if a.unmasked_only else keep
    batches = []
    for i in range(0, len(seqs), a.batch):
        batches.append(dict(input_ids=ids[i:i + a.batch].to(device), attention_mask=enc["attention_mask"][i:i + a.batch].to(device),
                            labels=labels[i:i + a.batch].to(device), keep=swap[i:i + a.batch].to(device),
                            masked=masked[i:i + a.batch].to(device)))

    l_orig = masked_loss(model, block, batches)
    nz = np.load(os.path.join(a.outdir, "norm.npz"))
    mean = torch.from_numpy(nz["mean"]).to(device)
    l_abl = masked_loss(model, block, batches, transform=lambda h: mean.expand_as(h))
    ts.log(f"original loss {l_orig:.4f}; mean-ablated loss {l_abl:.4f}")

    path = os.path.join(a.outdir, "loss_recovered_unmasked.json" if a.unmasked_only else "loss_recovered.json")
    res = load_json(path) if os.path.exists(path) else {}
    res.update(original=l_orig, mean_ablated=l_abl, n_windows=len(seqs), mask_frac=MASK_FRAC, seed=LOSS_SEED,
               layer=LOSS_LAYER, masked_tokens=int(masked.sum()))
    res.setdefault("runs", {})
    for name in a.runs:
        if name in res["runs"]:
            ts.log(f"{name}: already measured")
            continue
        sae, m, s, _ = load_sae(os.path.join(a.outdir, "runs", name), device)

        def tf(h, sae=sae, m=m, s=s):
            return sae((h - m) * s)[0] / s + m
        l = masked_loss(model, block, batches, transform=tf)
        res["runs"][name] = dict(loss=l, recovered=loss_recovered(l_abl, l, l_orig))
        ts.log(f"{name}: loss {l:.4f}, recovered {res['runs'][name]['recovered']:.4f}")
        ts.write_atomic_json(path, res)
    ts.write_atomic_json(path, res)


def cmd_summary(a):
    rows = []
    lr = load_json(os.path.join(a.outdir, "loss_recovered.json")) if os.path.exists(os.path.join(a.outdir, "loss_recovered.json")) else {"runs": {}}
    cross = load_json(os.path.join(a.outdir, "crossseed.json")) if os.path.exists(os.path.join(a.outdir, "crossseed.json")) else []
    by_run = {n: c for c in cross for n in c["runs"]}
    for rdir in finished_runs(a.outdir):
        name = os.path.basename(rdir)
        c = load_json(os.path.join(rdir, "config.json"))
        d = load_json(os.path.join(rdir, "DONE.json"))
        r = dict(name=name, kind=c["kind"], l1=c["l1"], k=c["k"], seed=c["seed"], steps=c["steps"],
                 passes=round(d["passes"], 2), matmul=c["matmul"], minutes=round(d["seconds"] / 60, 1))
        ep = os.path.join(rdir, "eval.json")
        if os.path.exists(ep):
            e = load_json(ep)
            r.update(fvu=e["fvu"], l0=e["l0"], inactive=e["inactive_frac"], always_on=e["always_on_frac"])
            r.update({f"fvu_{k}": v for k, v in e["fvu_by_class"].items()})
        if name in by_run:
            r.update(mean_max_cos=by_run[name]["mean_max_cos"], matched_frac=by_run[name]["matched_frac"])
        if name in lr["runs"]:
            r["loss_recovered"] = lr["runs"][name]["recovered"]
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(a.outdir, "summary.tsv"), sep="\t", index=False, float_format="%.5g")
    print(t.to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("heldout")
    p.add_argument("--outdir", required=True)
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("crossseed")
    p.add_argument("--outdir", required=True)
    p = sub.add_parser("loss")
    p.add_argument("--outdir", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--windows", nargs="+", required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--n-windows", type=int, default=1000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--unmasked-only", action="store_true")
    p = sub.add_parser("summary")
    p.add_argument("--outdir", required=True)
    a = ap.parse_args()
    {"heldout": cmd_heldout, "crossseed": cmd_crossseed, "loss": cmd_loss, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    main()
