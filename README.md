# FYP pipeline: benchmark kit

Everything here except the final GPU run works on a laptop with no GPU.

```
scripts/
  make\_corpus\_manifest.py   corpus promoter list + 500-window benchmark sample
  extract\_windows.py        windows from the reference FASTA, strand-aware
  benchmark.py              D3 benchmark: smoke, tokens, sweep, precision,
                            hooks, determinism, shards (resumable), project
data/                       manifests produced by make\_corpus\_manifest.py
```

## At home: do all of this before touching the lab PC

### 1\. Version control

Put this folder in a **private** GitHub repository. Home and lab then stay in
sync with `git pull` / `git push`. Do not commit the FASTA, model weights or
activation shards (see `.gitignore`).

### 2\. Environment

```
python -m venv .venv
.venv\\Scripts\\activate          # Windows   (Linux: source .venv/bin/activate)
pip install torch               # CPU build is fine at home
pip install -r requirements.txt
```

On the lab PC, install the CUDA build of PyTorch instead (the selector at
pytorch.org gives the command). PyTorch's pip wheels ship their own CUDA
runtime, so you do NOT install the CUDA toolkit. You only need the NVIDIA
driver to be recent enough, which `nvidia-smi` shows.

### 3\. Rebuild the manifests and compare

```
python scripts/make\_corpus\_manifest.py --sd6 41467\_2025\_63642\_MOESM8\_ESM.xlsx \\
    --covariates cov\_dry.tsv --groups gene\_groups\_id85.tsv --outdir data
```

Expected: corpus promoter component **45,117**, and these sha256 prefixes:
`corpus\_promoters.tsv d674162d…`, `benchmark\_sample.tsv 113d39fb…`.
A different hash means a different input file, so find out why before going on.

### 4\. Extract the 500 benchmark windows

```
python scripts/extract\_windows.py --manifest data/benchmark\_sample.tsv \\
    --fasta "E:\\genome data\\...\\161010\_Chinese\_Spring\_v1.0\_pseudomolecules.fasta" \\
    --out data/benchmark\_windows.fa
```

Expect `wrote 500 windows`, `coordinate mismatches against manifest: 0`.

### 5\. Download AgroNT at the pinned revision

```
huggingface-cli download InstaDeepAI/agro-nucleotide-transformer-1b \\
    --revision 9be6ae0d285053466d76ec50b042b8628065efd0 \\
    --local-dir models/agro-nt-1b
```

Copy `models/` to an external drive for the lab, so the lab PC needs no
internet and no download time.

### 6\. CPU smoke test: confirms the facts the scripts assume

```
python scripts/benchmark.py --model models/agro-nt-1b \\
    --windows data/benchmark\_windows.fa --device cpu --dtype bf16 \\
    --max-seqs 4 --stages smoke,tokens,hooks --strict
```

`--dtype bf16` keeps RAM use to a few GB. Check three things in the output:

* smoke: config matches (strict mode stops if not)
* tokens: real tokens per window, and whether an EOS token is added
* hooks: layers 4–30 IDENTICAL; whether layer 40 is DIFFERENT (final layer norm)

## At the lab: one GPU session

```
git pull
# activate env (see "keeping the environment" below)
nvidia-smi                       # record GPU, driver, CUDA
python scripts/benchmark.py --model <drive>/models/agro-nt-1b \\
    --windows data/benchmark\_windows.fa --stages all --outdir bench\_out
```

Then the restart test:

```
python scripts/benchmark.py ... --stages shards --simulate-kill-after 2 --outdir bench\_kill
python scripts/benchmark.py ... --stages shards --outdir bench\_kill
```

The second run must report the incomplete shard removed, earlier shards
verified, and `missing 0`. Commit `bench\_out/benchmark\_report.json`.

## Keeping the environment between lab sessions

In order of preference; which one works depends on the lab's rules.

1. A persistent user account on the lab PC. The venv lives in your home
directory and survives between sessions. Ask the lab admin first.
2. Everything on an external SSD: venv, model weights, data, activations.
Plug in, activate, run. Requires the lab PC to run the same OS as the SSD
was set up for.
3. Offline wheelhouse: at home run `pip download -r requirements.txt -d wheels`
(plus the CUDA torch wheel for the lab's OS and Python version), then at
the lab run `pip install --no-index --find-links wheels -r requirements.txt`.
Rebuilds in minutes with no internet.

