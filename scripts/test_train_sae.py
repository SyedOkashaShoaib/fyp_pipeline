"""
Unit tests for scripts/train_sae.py, on CPU, with tiny synthetic stores.

  cd scripts && python -m unittest test_train_sae
"""
import argparse
import json
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_sae as ts  # noqa: E402


def fake_store(root, name, n_win, seed, cls="promoter"):
    """n_win windows of 3 tokens, 4 windows to a shard, with known structure."""
    s = os.path.join(root, name)
    os.makedirs(s)
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(0, n_win, 4):
        k = min(4, n_win - i)
        arr = (rng.normal(size=(k * 3, ts.INPUT_DIM)) + 5).astype(np.float16)
        sh = f"shard_{i // 4:05d}.npy"
        np.save(os.path.join(s, sh), arr)
        for j in range(k):
            rows.append(dict(window_id=f"{name}_{i + j}", group_id=f"g{(i + j) // 2}_{name}",
                             region_class=cls, chrom="chr1A", shard=sh, row_start=j * 3, row_end=j * 3 + 3))
    pd.DataFrame(rows).to_csv(os.path.join(s, "windows_index.tsv"), sep="\t", index=False)
    with open(os.path.join(s, "manifest.jsonl"), "w") as f:
        f.write("{}\n")
    with open(os.path.join(s, "COMPLETE.json"), "w") as f:
        json.dump({"windows": n_win, "rows": n_win * 3}, f)
    return s


def run_args(outdir, **kw):
    d = dict(outdir=outdir, name="r", kind="relu", l1=0.01, k=0, seed=3, width=32, batch=8, passes=0, steps=20,
             lr=1e-3, grad_clip=1.0, l1_warmup=0.0, init_norm=1.0, chunk_shards=3, matmul="fp32", device="cpu", log_every=5, ckpt_minutes=99,
             kill_after=0)
    d.update(kw)
    return argparse.Namespace(**d)


class Small(unittest.TestCase):
    def test_heldout_keeps_groups_whole(self):
        w = pd.DataFrame({"group_id": [f"g{i // 3}" for i in range(3000)]})
        h = ts.assign_heldout(w)
        self.assertAlmostEqual(h.mean(), 0.10, delta=0.02)
        self.assertTrue((pd.Series(h).groupby(w["group_id"]).nunique() == 1).all())
        self.assertTrue((h == ts.assign_heldout(w)).all())

    def test_ranges_merge(self):
        self.assertEqual(ts.ranges([(0, 3), (3, 6), (9, 12)]), [[0, 6], [9, 12]])

    def test_topk_exact_sparsity(self):
        m = ts.SAE(16, 64, "topk", 5, generator=torch.Generator().manual_seed(0))
        f = m.encode(torch.randn(10, 16))
        self.assertTrue(((f > 0).sum(-1) <= 5).all())

    def test_relu_loss_has_l1_term(self):
        m = ts.SAE(16, 64, "relu", generator=torch.Generator().manual_seed(0))
        x = torch.randn(10, 16)
        total, rec, f = ts.sae_loss(m, x, 0.5)
        pen = (f * m.W_dec.norm(dim=1)).sum(-1).mean()
        self.assertAlmostEqual(total.item(), (rec + 0.5 * pen).item(), places=4)

    def test_lr_schedule(self):
        self.assertLess(ts.lr_at(0, 1000, 1.0), 0.2)
        self.assertEqual(ts.lr_at(500, 1000, 1.0), 1.0)
        self.assertAlmostEqual(ts.lr_at(900, 1000, 1.0), 0.5)
        self.assertEqual(ts.lr_at(1000, 1000, 1.0), 0.0)

    def test_l1_warmup_ramps_then_holds(self):
        self.assertAlmostEqual(ts.l1_at(0, 1000, 2.0, 0.1), 0.02)
        self.assertEqual(ts.l1_at(99, 1000, 2.0, 0.1), 2.0)
        self.assertEqual(ts.l1_at(500, 1000, 2.0, 0.1), 2.0)
        self.assertEqual(ts.l1_at(0, 1000, 2.0, 0.0), 2.0)

    def test_init_norm_sets_decoder_length(self):
        m = ts.SAE(16, 64, "relu", generator=torch.Generator().manual_seed(0), init_norm=0.1)
        self.assertTrue(torch.allclose(m.W_dec.norm(dim=1), torch.full((64,), 0.1)))

    def test_chunk_plan_covers_all(self):
        got = np.concatenate(ts.chunk_plan(17, 1, 0, 4))
        self.assertEqual(sorted(got.tolist()), list(range(17)))


class Pipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(lambda: setattr(ts, "INPUT_DIM", 1500))
        self.addCleanup(lambda: setattr(ts, "MONITOR_TOKENS", 65536))
        ts.INPUT_DIM = 12
        ts.MONITOR_TOKENS = 16
        self.stores = [fake_store(self.tmp.name, "a", 60, 1), fake_store(self.tmp.name, "b", 40, 2, "intron")]
        self.out = os.path.join(self.tmp.name, "out")
        ts.cmd_prepare(argparse.Namespace(stores=self.stores, outdir=self.out))

    def index(self):
        return json.load(open(os.path.join(self.out, "shard_index.json")))

    def test_prepare_accounts_for_every_token(self):
        p = json.load(open(os.path.join(self.out, "prep.json")))
        self.assertEqual(p["train_tokens"] + p["heldout_tokens"], 300)
        self.assertEqual(sum(s["rows"] for s in self.index()), p["train_tokens"])
        self.assertEqual(np.load(os.path.join(self.out, "heldout_tokens.npy")).shape, (p["heldout_tokens"], 12))

    def test_normalisation_is_from_training_rows(self):
        x = np.concatenate([ts.load_rows(s["path"], s["train"]) for s in self.index()]).astype(np.float64)
        nz = np.load(os.path.join(self.out, "norm.npz"))
        self.assertTrue(np.allclose(x.mean(0), nz["mean"], atol=1e-5))
        y = (x - nz["mean"]) * float(nz["scale"])
        self.assertAlmostEqual((y ** 2).sum(1).mean(), 12, places=3)

    def test_stream_is_deterministic_and_resumable(self):
        def take(start, n):
            out = []
            g = ts.batch_stream(self.index(), 5, 8, 3, start=start)
            for _ in range(n):
                p, c, b, x = next(g)
                out.append(((p, c, b), x.clone()))
            g.close()
            return out
        a = take((0, 0, 0), 12)
        pos = a[5][0]
        b = take((pos[0], pos[1], pos[2] + 1), 6)
        for (pa, xa), (pb, xb) in zip(a[6:], b):
            self.assertEqual(pa, pb)
            self.assertTrue(torch.equal(xa, xb))

    def test_training_learns_and_writes_outputs(self):
        ts.train_run(run_args(self.out, steps=60))
        d = os.path.join(self.out, "runs", "r")
        done = json.load(open(os.path.join(d, "DONE.json")))
        self.assertEqual(done["steps"], 60)
        self.assertFalse(os.path.exists(os.path.join(d, "resume.pt")))
        log = pd.read_csv(os.path.join(d, "log.tsv"), sep="\t")
        self.assertLess(log["mon_fvu"].iloc[-1], log["mon_fvu"].iloc[0])

    def test_kill_and_resume_matches_clean_run(self):
        ts.train_run(run_args(self.out, name="clean"))
        with self.assertRaises(SystemExit):
            ts.train_run(run_args(self.out, name="killed", kill_after=11))
        ts.train_run(run_args(self.out, name="killed"))
        a = torch.load(os.path.join(self.out, "runs", "clean", "final.pt"), weights_only=False)["state"]
        b = torch.load(os.path.join(self.out, "runs", "killed", "final.pt"), weights_only=False)["state"]
        for k in a:
            self.assertTrue(torch.equal(a[k], b[k]), k)

    def test_stop_file_saves_then_resume_matches_clean_run(self):
        ts.train_run(run_args(self.out, name="clean"))
        stop = os.path.join(self.out, "STOP")
        open(stop, "w").close()
        with self.assertRaisesRegex(SystemExit, "stopped"):
            ts.train_run(run_args(self.out, name="stopped"))
        os.remove(stop)
        ts.train_run(run_args(self.out, name="stopped"))
        a = torch.load(os.path.join(self.out, "runs", "clean", "final.pt"), weights_only=False)["state"]
        b = torch.load(os.path.join(self.out, "runs", "stopped", "final.pt"), weights_only=False)["state"]
        for k in a:
            self.assertTrue(torch.equal(a[k], b[k]), k)

    def test_passes_set_the_step_count(self):
        done = ts.train_run(run_args(self.out, name="p", steps=0, passes=2.0))
        self.assertEqual(done["steps"], 2 * 270 // 8)
        self.assertAlmostEqual(done["passes"], 2.0, delta=0.05)

    def test_resume_does_not_duplicate_log_rows(self):
        with self.assertRaises(SystemExit):
            ts.train_run(run_args(self.out, name="k", kill_after=11))
        ts.train_run(run_args(self.out, name="k"))
        steps = pd.read_csv(os.path.join(self.out, "runs", "k", "log.tsv"), sep="\t")["step"].tolist()
        self.assertEqual(steps, [5, 10, 15, 20])

    def test_trim_log_drops_later_and_partial_rows(self):
        p = os.path.join(self.tmp.name, "log.tsv")
        with open(p, "w") as f:
            f.write(ts.LOG_HEADER + "5\tx\n10\tx\n15\tx\n20\tpartial")
        ts.trim_log(p, 12)
        with open(p) as f:
            self.assertEqual(f.read(), ts.LOG_HEADER + "5\tx\n10\tx\n")

    def test_atomic_save_keeps_old_file_when_write_fails(self):
        p = os.path.join(self.tmp.name, "w.pt")
        ts.save_atomic_torch(p, {"a": 1})
        with self.assertRaises(Exception):
            ts.save_atomic_torch(p, {"a": lambda: None})
        self.assertEqual(torch.load(p, weights_only=False), {"a": 1})

    def test_prepare_skips_when_done_and_refuses_other_stores(self):
        before = os.path.getmtime(os.path.join(self.out, "norm.npz"))
        ts.cmd_prepare(argparse.Namespace(stores=self.stores, outdir=self.out))
        self.assertEqual(os.path.getmtime(os.path.join(self.out, "norm.npz")), before)
        with self.assertRaisesRegex(SystemExit, "different stores"):
            ts.cmd_prepare(argparse.Namespace(stores=self.stores[:1], outdir=self.out))

    def test_prepare_killed_before_prep_json_reruns_cleanly(self):
        os.remove(os.path.join(self.out, "prep.json"))
        ts.cmd_prepare(argparse.Namespace(stores=self.stores, outdir=self.out))
        self.assertTrue(os.path.exists(os.path.join(self.out, "prep.json")))

    def test_changed_settings_refused(self):
        with self.assertRaises(SystemExit):
            ts.train_run(run_args(self.out, name="x", kill_after=5))
        with self.assertRaisesRegex(SystemExit, "differ"):
            ts.train_run(run_args(self.out, name="x", l1=0.5))


if __name__ == "__main__":
    unittest.main()
