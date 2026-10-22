"""
Unit tests for scripts/extract_activations.py, on CPU, with a tiny randomly
initialised ESM model and a 6-mer tokenizer laid out like AgroNT's (4,105
model IDs; <eos> 4105 and <bos> 4106 exist only in the tokenizer).

  python -m unittest discover -s tests -v
"""
import itertools
import json
import os
import random
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(HERE, "..", "scripts")
sys.path.insert(0, SCRIPTS)
import benchmark  # noqa: E402
import extract_activations as ex  # noqa: E402

TINY = {"hidden_size": 16, "num_hidden_layers": 4, "num_attention_heads": 2,
        "vocab_size": 4105, "max_position_embeddings": 1026}
L = 1002


def build_model(path):
    from transformers import EsmConfig, EsmForMaskedLM, EsmTokenizer
    kmers = ["".join(p) for p in itertools.product("ACGT", repeat=6)]
    vocab = ["<unk>", "<pad>", "<mask>", "<cls>"] + kmers + list("ACGTN") + ["<eos>", "<bos>"]
    os.makedirs(path)
    with open(os.path.join(path, "vocab.txt"), "w") as f:
        f.write("\n".join(vocab) + "\n")
    tok = EsmTokenizer(os.path.join(path, "vocab.txt"), eos_token=None)
    tok.save_pretrained(path)
    torch.manual_seed(0)
    cfg = EsmConfig(intermediate_size=32, pad_token_id=1, mask_token_id=2,
                    token_dropout=False, emb_layer_norm_before=False,
                    position_embedding_type="absolute", **TINY)
    EsmForMaskedLM(cfg).save_pretrained(path)


def write_inputs(d, n, bad=None, name="w"):
    rnd = random.Random(1)
    fa, tsv = os.path.join(d, f"{name}.fa"), os.path.join(d, f"{name}.tsv")
    with open(fa, "w") as f, open(tsv, "w") as t:
        t.write("gene_id\tchrom\tstrand\ttss\tprom_start\tprom_end\tgc\tte_frac\tgroup_id\n")
        for i in range(n):
            chrom = "chrUn" if i % 5 == 4 else f"chr{1 + i % 7}{'ABD'[i % 3]}"
            strand = "+-"[i % 2]
            tss = 100000 + 5000 * i
            s, e = (tss - L, tss - 1) if strand == "+" else (tss + 1, tss + L)
            seq = "".join(rnd.choice("ACGT") for _ in range(L))
            if bad == i:
                seq = seq[:500] + "N" + seq[501:]
            gid = f"TraesCS1A02G{i:06d}"
            f.write(f">{gid} {chrom}:{s}-{e}({strand}) tss={tss} len={L}\n{seq}\n")
            t.write(f"{gid}\t{chrom}\t{strand}\t{tss}\t0\t0\t0.4\t0.0\tgrp{i // 2}\n")
    return fa, tsv


class ExtractorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.model = os.path.join(cls.tmp.name, "tiny")
        build_model(cls.model)
        cls.fa, cls.tsv = write_inputs(cls.tmp.name, 23)
        benchmark.EXPECTED = ex.benchmark.EXPECTED = dict(TINY)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.out = tempfile.mkdtemp(dir=self.tmp.name)

    def args(self, layers="2", extra=()):
        return ["--model", self.model, "--windows", self.fa, "--manifest", self.tsv,
                "--no-exclusion-check", "--region-class", "promoter", "--layers", layers,
                "--outdir", self.out, "--batch", "4", "--shard-windows", "5",
                "--reserve-gb", "0", "--device", "cpu", *extra]

    def run_cli(self, layers="2", extra=()):
        """Separate process, so a simulated kill doesn't end the test run."""
        code = ("import sys, benchmark, extract_activations as ex;"
                f"benchmark.EXPECTED = ex.benchmark.EXPECTED = {TINY!r};"
                "sys.exit(ex.main(sys.argv[1:]))")
        return subprocess.run([sys.executable, "-c", code, *self.args(layers, extra)],
                              cwd=SCRIPTS, capture_output=True, text=True)

    def reference(self, layer):
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(self.model)
        m = AutoModel.from_pretrained(self.model, add_pooling_layer=False).eval()
        seqs = [r["seq"] for r in ex.read_windows(self.fa)]
        with torch.no_grad():
            enc = tok(seqs, return_tensors="pt", padding=True)
            hs = m(**enc, output_hidden_states=True).hidden_states[layer]
        keep = benchmark.real_token_mask(tok, enc)
        return np.concatenate([hs[j][keep[j]].numpy() for j in range(len(seqs))])

    def store(self, layer):
        ldir = os.path.join(self.out, f"L{layer}")
        with open(os.path.join(ldir, "manifest.jsonl")) as f:
            recs = sorted((json.loads(x) for x in f), key=lambda r: r["shard"])
        arr = np.concatenate([np.load(os.path.join(ldir, r["file"])) for r in recs])
        return ldir, recs, arr

    # ------------------------------------------------------------- happy path

    def test_values_fields_and_completion(self):
        self.assertEqual(ex.main(self.args()), 0)
        ldir, recs, arr = self.store(2)
        self.assertEqual(arr.dtype, np.float16)
        self.assertEqual(arr.shape, (23 * 167, 16))
        np.testing.assert_allclose(arr.astype(np.float32), self.reference(2),
                                   rtol=2e-3, atol=2e-3)
        w = recs[3]["windows"][4]  # window 19, a chrUn window
        self.assertEqual((w["window_id"], w["subgenome"], w["n_tokens"], w["row_start"]),
                         ("TraesCS1A02G000019", "Un", 167, 668))
        for f in ("gene_id", "group_id", "region_class", "chrom", "strand", "start", "end",
                  "tss", "length_bp", "seq_sha256", "token_bp", "row_end"):
            self.assertIn(f, w)
        done = json.load(open(os.path.join(ldir, "COMPLETE.json")))
        self.assertEqual((done["windows"], done["missing"], done["shards"]), (23, 0, 5))
        info = json.load(open(os.path.join(ldir, "run_info.json")))
        self.assertEqual((info["layer"], info["compute_dtype"], info["storage_dtype"]),
                         (2, "float32", "float16"))
        self.assertTrue(info["model_files_sha256"])
        with open(os.path.join(ldir, "windows_index.tsv"), "rb") as f:
            body = f.read()
        self.assertNotIn(b"\r\n", body)
        self.assertEqual(body.count(b"\n"), 24)

    def test_two_layers_in_one_pass(self):
        self.assertEqual(ex.main(self.args("3,1")), 0)
        for k in (1, 3):
            _, _, arr = self.store(k)
            np.testing.assert_allclose(arr.astype(np.float32), self.reference(k),
                                       rtol=2e-3, atol=2e-3)

    def test_rerun_of_a_complete_store_writes_nothing(self):
        ex.main(self.args())
        before = {f: os.path.getmtime(os.path.join(self.out, "L2", f))
                  for f in os.listdir(os.path.join(self.out, "L2")) if f.endswith(".npy")}
        self.assertEqual(ex.main(self.args()), 0)
        after = {f: os.path.getmtime(os.path.join(self.out, "L2", f)) for f in before}
        self.assertEqual(before, after)

    # ----------------------------------------------------------------- resume

    def test_kill_and_resume_matches_a_clean_run(self):
        r = self.run_cli(extra=("--simulate-kill-after", "2"))
        self.assertEqual(r.returncode, 9, r.stdout + r.stderr)
        self.assertTrue(os.path.exists(os.path.join(self.out, "L2", "shard_00002.npy.tmp")))
        r = self.run_cli()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("removed incomplete shard", r.stdout)
        self.assertIn("2 shards verified", r.stdout)
        _, recs, _ = self.store(2)
        resumed = [x["sha256"] for x in recs]
        self.out = tempfile.mkdtemp(dir=self.tmp.name)
        ex.main(self.args())
        _, recs, _ = self.store(2)
        self.assertEqual(resumed, [x["sha256"] for x in recs])

    def test_resume_adds_a_layer_and_fills_only_its_gaps(self):
        self.assertEqual(self.run_cli("2", ("--simulate-kill-after", "3")).returncode, 9)
        r = self.run_cli("2,3")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("layers [2, 3]", r.stdout)
        self.assertIn("layers [3]", r.stdout)
        for k in (2, 3):
            _, _, arr = self.store(k)
            self.assertEqual(arr.shape[0], 23 * 167)

    def test_orphan_shard_and_torn_manifest_line_are_cleaned(self):
        self.assertEqual(self.run_cli(extra=("--simulate-kill-after", "1")).returncode, 9)
        ldir = os.path.join(self.out, "L2")
        os.replace(os.path.join(ldir, "shard_00001.npy.tmp"), os.path.join(ldir, "shard_00001.npy"))
        with open(os.path.join(ldir, "manifest.jsonl"), "a") as f:
            f.write('{"shard": 1, "file": "shard_000')
        r = self.run_cli()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("dropped a half-written last manifest line", r.stdout)
        self.assertIn("never reached the manifest", r.stdout)

    def test_corrupt_shard_stops_resume(self):
        ex.main(self.args())
        with open(os.path.join(self.out, "L2", "shard_00001.npy"), "r+b") as f:
            f.seek(200)
            f.write(b"\x00\x01")
        with self.assertRaisesRegex(SystemExit, "checksum failure"):
            ex.main(self.args())

    def test_changed_settings_are_refused(self):
        ex.main(self.args())
        with self.assertRaisesRegex(SystemExit, "shard_windows"):
            ex.main(self.args(extra=("--shard-windows", "7")))
        with self.assertRaisesRegex(SystemExit, "region_class"):
            a = self.args()
            a[a.index("promoter")] = "intron"
            ex.main(a)

    # ----------------------------------------------------------------- guards

    def test_n_window_is_refused_before_writing(self):
        fa, tsv = write_inputs(self.tmp.name, 6, bad=3, name="withN")
        a = self.args()
        a[a.index(self.fa)], a[a.index(self.tsv)] = fa, tsv
        with self.assertRaisesRegex(SystemExit, "non-ACGT"):
            ex.main(a)
        self.assertEqual(os.listdir(self.out), [])

    def test_excluded_gene_is_refused(self):
        exc = os.path.join(self.out, "exclusion.tsv")
        with open(exc, "w") as f:
            f.write("gene_id\treason\nTraesCS1A02G000007\tevaluation_gene\n")
        a = self.args()
        a[a.index("--no-exclusion-check")] = "--exclusion=" + exc
        with self.assertRaisesRegex(SystemExit, "excluded genes"):
            ex.main(a)

    def test_exclusion_choice_is_required(self):
        a = self.args()
        a.remove("--no-exclusion-check")
        with self.assertRaisesRegex(SystemExit, "--exclusion"):
            ex.main(a)

    def test_content_hash_mismatch_is_refused(self):
        with self.assertRaisesRegex(SystemExit, "content hash"):
            ex.main(self.args(extra=("--expect-content-hash", "0000000000000000")))

    def test_layers_0_and_last_are_refused(self):
        for bad in ("0", "4"):
            with self.assertRaisesRegex(SystemExit, "not allowed"):
                ex.main(self.args(bad))

    def test_config_mismatch_is_refused(self):
        benchmark.EXPECTED["hidden_size"] = 1500
        try:
            with self.assertRaisesRegex(SystemExit, "configuration differs"):
                ex.main(self.args())
        finally:
            benchmark.EXPECTED["hidden_size"] = 16

    def test_storage_reserve_is_enforced(self):
        a = self.args()
        a[a.index("--reserve-gb") + 1] = "1e9"
        with self.assertRaisesRegex(SystemExit, "not enough free space"):
            ex.main(a)
        self.assertFalse(any(f.endswith(".npy") for f in os.listdir(os.path.join(self.out, "L2"))))

    def test_dry_run_writes_nothing(self):
        self.assertEqual(ex.main(self.args(extra=("--dry-run",))), 0)
        self.assertEqual(os.listdir(self.out), [])

    def test_fp16_overflow_and_nonfinite_are_refused(self):
        with self.assertRaisesRegex(SystemExit, "float16"):
            ex.check_activations(torch.tensor([[1.0, 70000.0]]), "w", 13)
        with self.assertRaisesRegex(SystemExit, "non-finite"):
            ex.check_activations(torch.tensor([[float("nan")]]), "w", 13)
        ex.check_activations(torch.tensor([[65504.0, -65504.0]]), "w", 13)

    def test_token_id_outside_model_vocabulary_is_refused(self):
        with self.assertRaisesRegex(SystemExit, "outside the model vocabulary"):
            ex.check_ids(torch.tensor([[3, 4105]]), 4105)

    # ---------------------------------------------------------------- helpers

    def test_token_span_and_subgenome(self):
        plus = {"strand": "+", "start": 1001, "end": 2002}
        minus = {"strand": "-", "start": 1001, "end": 2002}
        self.assertEqual(ex.token_span(plus, 0), (1001, 1006))
        self.assertEqual(ex.token_span(plus, 166), (1997, 2002))
        self.assertEqual(ex.token_span(minus, 0), (1997, 2002))
        self.assertEqual(ex.token_span(minus, 166), (1001, 1006))
        self.assertEqual([ex.subgenome(c) for c in ("chr1A", "chr7D", "chrUn")], ["A", "D", "Un"])
        with self.assertRaises(SystemExit):
            ex.subgenome("chr1n")


if __name__ == "__main__":
    unittest.main()
