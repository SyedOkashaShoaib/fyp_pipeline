# AgroNT activation-extraction GPU feasibility benchmark

## Status

**Outcome:** The allocated laboratory PC passed the model-loading, tokenization,
activation-capture, determinism, numerical-precision, throughput, and GPU-memory
checks performed in this benchmark.

This report documents what was run, why each check was performed, what was
observed, and how another person can repeat the work. It describes the benchmark
artifacts committed in repository commit
[`4135e2e`](https://github.com/SyedOkashaShoaib/fyp_pipeline/commit/4135e2e91b2499e685044e5b068794ce3a610785).

The completed work establishes the feasibility of **AgroNT forward inference and
activation extraction** on the allocated GPU. It does not yet establish the
runtime or memory requirements of sparse-autoencoder training or the complete
downstream biological analysis.

## 1. Why this benchmark was needed

The project will pass wheat promoter sequences through a frozen AgroNT model and
save an internal activation for later sparse-autoencoder analysis. Before
performing this at full scale, we needed evidence that the laboratory PC could:

1. load the correct model architecture;
2. tokenize 1,000-bp promoter windows as expected;
3. capture a selected internal layer correctly;
4. produce repeatable activations;
5. run without numerical failure in the precisions under consideration;
6. process sequences at a practical rate; and
7. remain comfortably within GPU-memory limits.

The benchmark therefore answers a feasibility question: **Can the allocated PC
reliably run AgroNT activation extraction at the scale anticipated by the
project?**

## 2. Recorded environment

| Component | Recorded value |
|---|---|
| Operating system | Windows 11 (`10.0.26200`) |
| Python | 3.13.0 |
| PyTorch | 2.14.1+cu132 |
| Transformers | 4.46.3 |
| GPU | NVIDIA RTX 4000 Ada Generation |
| GPU memory | 19.99 GB |
| NVIDIA driver | 596.71 |
| Driver-reported CUDA compatibility | 13.2 |
| BF16 reported as supported | Yes |
| Benchmark computation precision | FP32 |

The benchmark reports record the model revision as:

```text
9be6ae0d285053466d76ec50b042b8628065efd0
```

The model was loaded from the local directory `models/agro-nt-1b`. Passing a
revision argument alongside a local directory does not independently prove the
identity of the local weights. A fully audited run should either download the
model directly at the pinned revision or record hashes for the local model
files.

## 3. Benchmark input construction

### 3.1 Corpus manifest

The manifest-generation script was run using:

```powershell
python scripts\make_corpus_manifest.py `
    --sd6 ".\data\SD6.xlsx" `
    --covariates ".\data\cov_dry.tsv" `
    --groups ".\data\gene_groups_id85.tsv" `
    --outdir "data"
```

The script reported:

| Output | Result |
|---|---:|
| Evaluation genes | 41,267 |
| Related-gene groups represented by the evaluation set | 18,901 |
| Corpus promoter component | 45,117 |
| Benchmark sample | 500 promoters |
| Subgenome A | 154 |
| Subgenome B | 162 |
| Subgenome D | 161 |
| Other/generated label `n` | 23 |

The 500-row sample total is internally consistent: `154 + 162 + 161 + 23 =
500`. The meaning of the 23 `n` assignments should be audited before the
dataset is declared final.

The newly generated manifest hashes differed from prefixes previously written
in the repository README. This discrepancy was deliberately deferred during
the compute-feasibility session. It does not prevent a GPU timing test, but it
must be resolved before the benchmark dataset is treated as the frozen official
dataset.

### 3.2 Promoter-window extraction

The 500 genomic windows were extracted with:

```powershell
python scripts\extract_windows.py `
    --manifest ".\data\benchmark_sample.tsv" `
    --fasta "<PATH_TO_IWGSC_REFSEQ_V1_FASTA>" `
    --out ".\data\benchmark_500.fasta" `
    --length 1000
```

Observed result:

```text
wrote 500 windows
skipped 0 at contig edges
coordinate mismatches against manifest: 0
sequence-content sha256 prefix: e9e9793b0f3d0fb1…
```

Each input sequence is a 1,000-base promoter window. AgroNT uses 6-mer tokens.
For a 1,000-base sequence, the tokenizer produced 166 complete 6-mers, four
single-nucleotide remainder tokens, and one `<cls>` token:

```text
166 six-mers + 4 single bases + 1 <cls> = 171 tokens
```

No promoter bases were discarded to make the sequence divisible by six.

## 4. Tests performed and why

### 4.1 Smoke test

**Purpose:** Verify that the local checkpoint loads and that its architecture
matches the assumptions made by the extraction code.

Command:

```powershell
python scripts\benchmark.py `
    --model ".\models\agro-nt-1b" `
    --revision "9be6ae0d285053466d76ec50b042b8628065efd0" `
    --windows ".\data\benchmark_500.fasta" `
    --device cuda `
    --dtype fp32 `
    --stages "smoke,tokens" `
    --max-seqs 8 `
    --batch-sizes "1" `
    --outdir ".\bench_out\smoke_test"
```

Results:

| Property | Result |
|---|---:|
| Architecture | `EsmForMaskedLM` |
| Parameters | 991,261,206 |
| Hidden size | 1,500 |
| Transformer blocks | 40 |
| Attention heads | 20 |
| Vocabulary size | 4,105 |
| Maximum position embeddings | 1,026 |
| Expected configuration matched | Yes |

### 4.2 Tokenization check

**Purpose:** Confirm the actual number and layout of tokens produced from the
project's 1,000-bp promoter windows. Token count affects both memory and
throughput.

Results over the eight tested windows:

| Property | Result |
|---|---:|
| Minimum tokens per window | 171 |
| Median tokens per window | 171 |
| Maximum tokens per window | 171 |
| `<cls>` token | Present |
| EOS token | Absent |
| Single-nucleotide remainder tokens in window 1 | 4 |
| Windows containing `N` | 0 of 8 |

### 4.3 Hook-equivalence check

**Purpose:** Confirm that a forward hook on a transformer block captures the
same tensor as the corresponding item returned in `hidden_states`. This permits
future extraction of one selected layer without retaining every hidden layer.

Command:

```powershell
python scripts\benchmark.py `
    --model ".\models\agro-nt-1b" `
    --revision "9be6ae0d285053466d76ec50b042b8628065efd0" `
    --windows ".\data\benchmark_500.fasta" `
    --device cuda `
    --dtype fp32 `
    --stages "hooks,determinism" `
    --layers "4,12,20,30,40" `
    --shard-layer 20 `
    --max-seqs 8 `
    --batch-sizes "1" `
    --strict `
    --outdir ".\bench_out\validation_test"
```

Results:

| Layer | Maximum absolute difference | Identical? |
|---:|---:|---|
| 4 | 0 | Yes |
| 12 | 0 | Yes |
| 20 | 0 | Yes |
| 30 | 0 | Yes |
| 40 | 2,852.5623 | No |

The layer-40 difference is expected. The hook captures the final transformer
block output before the model's final layer normalization, whereas
`hidden_states[40]` contains the post-normalization tensor. Layer 20—the
provisional extraction site—was exactly identical.

### 4.4 Determinism check

**Purpose:** Verify that the same inputs produce the same activations on two
successive evaluation-mode forward passes.

For layers 4, 12, 20, 30, and 40, the maximum absolute difference between the
two runs was exactly zero.

### 4.5 Reduced-precision comparison

**Purpose:** Compare BF16 and FP16 computation against an FP32 reference and
detect non-finite outputs or unacceptable numerical drift.

The comparison used 32 promoter windows and layers 4, 12, 20, 30, and 40. All
outputs were finite.

At the provisional layer 20:

| Computation | Mean cosine similarity | Minimum cosine similarity | Mean relative error | Maximum relative error |
|---|---:|---:|---:|---:|
| BF16 | 0.999939 | 0.990992 | 1.1315% | 16.4093% |
| FP16 | 0.999999 | 0.999943 | 0.1432% | 1.3066% |

FP16 was closer to FP32 than BF16 for every tested layer in this sample.
Nevertheless, the timing benchmark retained FP32 as the reference computation
precision. Using FP16 for stored activations is a separate decision from using
FP16 for the model's forward computation.

### 4.6 Pilot batch-size sweep

**Purpose:** Find a practical batch size without immediately attempting the
largest configured batch.

The pilot used 64 promoters in FP32:

| Batch size | Windows/s | Tokens/s | Peak allocated GPU memory |
|---:|---:|---:|---:|
| 1 | 22.9490 | 3,927.98 | 3.829 GB |
| 2 | 27.0060 | 4,622.54 | 3.901 GB |
| 4 | 29.2984 | 5,015.26 | 3.982 GB |
| 8 | 29.3848 | 5,024.79 | 4.207 GB |
| 16 | 30.2626 | 5,174.90 | 4.657 GB |
| 32 | 30.7180 | 5,257.57 | 5.567 GB |

Throughput was beginning to plateau by batch size 32, while memory use remained
well below the available 19.99 GB. Batch size 32 was therefore selected for the
repeated timing runs.

### 4.7 Three repeated timing runs

**Purpose:** Measure full 500-window throughput and determine whether performance
is stable across repeated runs.

Each run used FP32, batch size 32, two warm-up iterations, strict architecture
checking, and a separate output directory.

| Run | Windows/s | Tokens/s | Peak allocated GPU memory |
|---:|---:|---:|---:|
| 1 | 31.018622 | 5,309.7767 | 6.1074 GB |
| 2 | 30.803578 | 5,272.9654 | 6.1074 GB |
| 3 | 30.595879 | 5,237.4114 | 6.1074 GB |
| **Mean** | **30.806027** | **5,273.3845** | **6.1074 GB** |

Derived stability measures:

| Measure | Result |
|---|---:|
| Sample standard deviation | 0.2114 windows/s |
| Coefficient of variation | 0.686% |
| Fastest-to-slowest range as a percentage of the mean | 1.372% |

The three runs were therefore highly consistent.

## 5. Runtime projection

Using the measured mean of 30.8060 windows/s:

| Number of promoter windows | Forward-pass projection |
|---:|---:|
| 500 | 16.23 seconds |
| 45,117 | 24.41 minutes |
| 80,000 | 43.28 minutes |
| 150,000 | 81.15 minutes |

These values project model-forward time from the measured throughput. They do
not include model loading, tokenization outside the timed region, activation
serialization, checksum generation, or other pipeline overhead.

## 6. Interpretation

The allocated NVIDIA RTX 4000 Ada Generation GPU has sufficient memory and
throughput for the tested AgroNT activation-extraction workload:

- The model loaded successfully with the expected architecture.
- All tested 1,000-bp windows tokenized to a consistent length.
- The provisional layer-20 hook captured the exact expected activation.
- Repeated inference was deterministic in the tested configuration.
- FP32 inference processed approximately 30.8 promoter windows per second.
- Peak PyTorch-allocated GPU memory was approximately 6.11 GB out of 19.99 GB.
- Three full timing runs showed less than 0.7% coefficient of variation.

Therefore, **GPU capacity is not currently a blocker for AgroNT forward
inference and activation extraction**.

## 7. Important scope and limitations

1. **The sweep is conservative but not identical to the intended extractor.**
   The current benchmark calls the masked-language-model class with
   `output_hidden_states=True`. It therefore calculates the MLM output and
   retains all hidden states. The planned production extractor should run the
   base transformer and capture only the selected activation site. The measured
   workload is heavier, but its timing is not an exact measurement of that
   future implementation.

2. **The local checkpoint identity was not independently hashed.** The reports
   record the intended revision, but the model was loaded from a local folder.

3. **Manifest hashes remain unresolved.** The regenerated manifests did not
   match hash prefixes previously recorded in the README. This must be resolved
   before the input dataset is frozen for official analysis.

4. **The 23 `n` subgenome labels require explanation.** They did not obstruct
   compute benchmarking but require a data audit.

5. **Shard writing, interruption recovery, and the `project` stage were not
   completed.** No claim is made that restartability or end-to-end activation
   serialization has passed.

6. **HDD versus SSD performance was not measured.** The sweep loads the FASTA
   into memory before timing inference, so moving only the input FASTA would not
   materially change the reported model throughput. Output-drive performance
   is more relevant when large activation shards are written.

7. **SAE training was not benchmarked.** This report covers AgroNT inference and
   activation-extraction feasibility only.

## 8. Reproduction procedure

1. Clone the repository and check out the documented source revision.
2. Create and activate a virtual environment.
3. Install the CUDA-enabled PyTorch build and the versions recorded in
   `env_lock_lab.txt`.
4. Obtain the pinned AgroNT checkpoint and verify its revision or file hashes.
5. Rebuild the corpus manifests from the documented source inputs.
6. Resolve any manifest-hash difference before declaring the input frozen.
7. Extract the 500 promoter windows from the matching IWGSC RefSeq v1.0 FASTA.
8. Confirm 500 windows, zero edge skips, and zero coordinate mismatches.
9. Run the smoke, token, hook, determinism, and precision stages.
10. Run the pilot sweep and select a batch size using both throughput and memory.
11. Run the full timing sweep three times using different output directories.
12. Calculate the mean, sample standard deviation, coefficient of variation,
    memory maximum, and runtime projections.

To repeat a full timing run:

```powershell
python scripts\benchmark.py `
    --model ".\models\agro-nt-1b" `
    --revision "9be6ae0d285053466d76ec50b042b8628065efd0" `
    --windows ".\data\benchmark_500.fasta" `
    --device cuda `
    --dtype fp32 `
    --stages "sweep" `
    --max-seqs 500 `
    --batch-sizes "32" `
    --warmup 2 `
    --strict `
    --outdir ".\bench_out\<UNIQUE_RUN_NAME>"
```

## 9. Committed evidence

The following reports are present under `bench_out/` in commit `4135e2e`:

- `smoke_test/benchmark_report.json`
- `validation_test/benchmark_report.json`
- `precision_test/benchmark_report.json`
- `pilot_sweep_fp32/benchmark_report.json`
- `timing_fp32_run1/benchmark_report.json`
- `timing_fp32_run2/benchmark_report.json`
- `timing_fp32_run3/benchmark_report.json`

These JSON files are the machine-readable evidence. This Markdown document is
their explanatory companion and should be updated if the benchmark is rerun or
if the extraction implementation changes.
