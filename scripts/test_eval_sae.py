"""
Unit tests for scripts/eval_sae.py, on CPU.

  cd scripts && python -m unittest test_eval_sae
"""
import os
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_sae as ev  # noqa: E402
import train_sae as ts  # noqa: E402


class Metrics(unittest.TestCase):
    def test_perfect_reconstruction_has_zero_fvu(self):
        class Identity(torch.nn.Module):
            W_dec = torch.zeros(4, 8)

            def forward(self, x):
                f = torch.zeros(len(x), 4)
                f[:, 0] = 1.0
                return x, f
        tokens = np.random.default_rng(0).normal(size=(40, 8)).astype(np.float16)
        labels = np.array([0, 1] * 20)
        r, freq = ev.heldout_metrics(Identity(), tokens, labels, 2, torch.zeros(8), 1.0, "cpu")
        self.assertAlmostEqual(r["fvu"], 0.0)
        self.assertEqual(r["l0"], 1.0)
        self.assertEqual(r["inactive_frac"], 0.75)
        self.assertEqual(r["always_on_frac"], 0.25)
        self.assertEqual(freq.tolist(), [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(r["tokens_by_class"], [20, 20])

    def test_zero_rebuild_has_fvu_one(self):
        class Zero(torch.nn.Module):
            W_dec = torch.zeros(4, 8)

            def forward(self, x):
                return torch.zeros_like(x), torch.zeros(len(x), 4)
        tokens = np.random.default_rng(0).normal(size=(40, 8)).astype(np.float16)
        r, _ = ev.heldout_metrics(Zero(), tokens, np.zeros(40, int), 1, torch.zeros(8), 2.0, "cpu")
        self.assertAlmostEqual(r["fvu"], 1.0)

    def test_best_cos_finds_permuted_copy(self):
        a = torch.randn(50, 16)
        b = a[torch.randperm(50)] * 3
        self.assertTrue(torch.allclose(ev.best_cos(a, b, chunk=7), torch.ones(50), atol=1e-5))

    def test_loss_recovered_scale(self):
        self.assertEqual(ev.loss_recovered(5.0, 1.0, 1.0), 1.0)
        self.assertEqual(ev.loss_recovered(5.0, 5.0, 1.0), 0.0)
        self.assertEqual(ev.loss_recovered(5.0, 2.0, 1.0), 0.75)

    def test_masks_fixed_rate_and_real_tokens_only(self):
        keep = np.ones((5, 168), bool)
        keep[:, 0] = False
        a = ev.make_masks(keep, 0.15, 1)
        self.assertTrue((a == ev.make_masks(keep, 0.15, 1)).all())
        self.assertFalse(a[:, 0].any())
        self.assertEqual(a.sum(1).tolist(), [25] * 5)

    def test_fasta_subset(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.fa")
            with open(p, "w") as f:
                f.write(">a chr1A:1-6(+) tss=0 len=6\nACG\nTAC\n>b chr1A:9-14(+) tss=0 len=6\nTTTTTT\n")
            self.assertEqual(ev.read_fasta_subset([p], ["b", "a"]), {"a": "ACGTAC", "b": "TTTTTT"})
            with self.assertRaises(SystemExit):
                ev.read_fasta_subset([p], ["c"])


class Hook(unittest.TestCase):
    """The hook swaps real-token vectors only; a transform that returns its input changes nothing."""

    def setUp(self):
        torch.manual_seed(0)
        self.block = torch.nn.Linear(4, 4)
        self.head = torch.nn.Linear(4, 6)
        outer = self

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Embedding(6, 4)
                self.block = outer.block
                self.head = outer.head

            def forward(self, input_ids, attention_mask):
                class O:
                    logits = self.head(self.block(self.emb(input_ids)))
                return O
        self.m = M()
        ids = torch.randint(0, 6, (3, 5))
        keep = torch.ones(3, 5, dtype=torch.bool)
        keep[:, 0] = False
        masked = torch.zeros(3, 5, dtype=torch.bool)
        masked[:, 2] = True
        self.batches = [dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids, keep=keep, masked=masked)]

    def test_identity_transform_matches_no_transform(self):
        a = ev.masked_loss(self.m, self.block, self.batches)
        b = ev.masked_loss(self.m, self.block, self.batches, transform=lambda h: h)
        self.assertAlmostEqual(a, b, places=5)

    def test_transform_changes_loss_and_hook_is_removed(self):
        a = ev.masked_loss(self.m, self.block, self.batches)
        b = ev.masked_loss(self.m, self.block, self.batches, transform=torch.zeros_like)
        self.assertNotAlmostEqual(a, b, places=3)
        self.assertEqual(len(self.block._forward_hooks), 0)


class Roundtrip(unittest.TestCase):
    def test_saved_sae_reloads_identically(self):
        with tempfile.TemporaryDirectory() as d:
            m = ts.SAE(8, 32, "topk", 3, generator=torch.Generator().manual_seed(0))
            cfg = dict(width=32, kind="topk", k=3)
            torch.save({"state": m.state_dict(), "config": cfg, "mean": torch.zeros(8), "scale": 1.5},
                       os.path.join(d, "final.pt"))
            n, mean, scale, c = ev.load_sae(d, "cpu")
            x = torch.randn(4, 8)
            self.assertTrue(torch.equal(m(x)[0], n(x)[0]))
            self.assertEqual(scale, 1.5)


if __name__ == "__main__":
    unittest.main()
