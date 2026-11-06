# SAE pilot protocol (deliverable D6)

Written 2026-11-04, before any SAE was trained on the corpus, and accepted by
Kumail on 2026-11-04 (class definitions and counts, 5-pass budget). The
settings below are frozen before the grid starts.

## Scope

The pilot trains ReLU/L1 and TopK sparse autoencoders on AgroNT layer 13
(`hidden_states[13]`, the output of `encoder.layer[12]`) and uses no response
label at any point (proposal sec. 6.4). The scripts read activation stores and
corpus windows only. They never open SD6, the splits file or any drought file.

## Corpus

Every window is 1,002 bp (167 tokens) and N-free. Built by
`scripts/build_corpus_regions.py`, seed 20261104, content hash
`564adbfb7e3297f0`.

| Class | Windows | Definition |
|---|---|---|
| promoter | 44,137 | `data/corpus_promoters_L1002.tsv`, unchanged |
| intron | 9,204 | Centred in an intron >= 1,002 bp of a corpus gene's .1 transcript; one per intron; gene strand |
| utr5_tss | 10,000 | Starts at the TSS of a corpus gene with an annotated 5' UTR, runs downstream; gene strand |
| splice | 10,000 | Centred on one exon-intron boundary; one per gene; gene strand |
| intergenic | 15,002 | >= 2 kb from any HC or LC gene, placed chromosomes, matched to promoters on subgenome x TE-fraction bin x GC decile; + strand |
| repeat | 15,000 | Inside one repeat_region whose dominant superfamily is >= 80%, >= 2 kb from any gene, placed chromosomes, equal allocation across superfamilies; element strand |

Total 103,343 windows, 17,258,281 tokens. The proposal gives the classes and the
80,000-150,000 range only (sec. 6.2); these definitions and counts are this
protocol's choice.

Leakage: genic classes use corpus genes only (HC genes not in
`exclusion_manifest.tsv`). No window overlaps the body or 1,002 bp flanks of an
evaluation gene or a member of its group. No window overlaps another corpus
window or repeats a sequence already in the corpus.

Known limits: the 5' UTR is short (median 121 bp), so `utr5_tss` windows mostly
hold what follows it. Repeat copies cannot be screened against evaluation
promoters by coordinate. The 12,853 Ensembl non-coding genes in the grouping
have no coordinates in the IWGSC files, so they cannot be masked or used. The
promoter store holds 113 repeated sequences (44,024 distinct); these are kept.

## Data preparation (`train_sae.py prepare`)

- Held out: 10% of group_ids, seed 20261105. Whole groups go together across
  classes. Intergenic and repeat windows are their own groups.
- Normalisation, from training windows only: subtract the per-dimension mean,
  then multiply by one global scale so that E||x||^2 = 1,500.
- Every gate is measured on held-out windows.

## Training

| Setting | Value |
|---|---|
| Width | 6,000 (4x) |
| Batch | 4,096 tokens |
| Budget | 5 passes over the training tokens (about 19,000 steps, about 78M tokens) |
| Optimiser | Adam, lr 2e-4, betas (0.9, 0.999), gradient clip 1.0 |
| Schedule | Linear warm-up over 1% of steps, linear decay to 0 over the last 20% |
| Mixing | Shards in a random order per pass, tokens shuffled within 8 shards |
| Decoder | ReLU: free, L1 term weighted by decoder norm, rows start at length 0.1. TopK: unit norm after each step |
| Resampling | None |
| Checkpoint | Weights, optimiser and position every 2 minutes, atomic; a `STOP` file or Ctrl+C saves and exits |
| Seeds | 3 per setting: 1, 2, 3 |

Grid (`docs/sae_pilot_grid.json`): ReLU/L1 at lambda = 1.5, 3, 5, 7, 15; TopK at k = 32, 64, 128.
24 runs. Lambda 1.5 is an extension beyond the 4 values first planned (see smoke-test results).

## Rules fixed before training

1. **Lambda.** Short single-seed runs at lambda 0.5, 1, 2, 5, 10 (3,000 steps).
   Pick 4 values whose held-out L0 spans about 20 to 200, extending the range if
   needed. Rerunning this scan does not look at any gate other than L0.
2. **Precision.** Matrix multiplies run in TF32 if, on one seed and one setting,
   fp32 and TF32 differ in held-out FVU, L0 and inactive rate by less than the
   spread across seeds 1, 2, 3 at that setting. Otherwise the grid runs in fp32.
3. **Convergence.** A run has converged when held-out FVU improved by less than
   1% of its value over the last pass-equivalent of the constant-lr phase. Any
   run that has not converged is reported as such. If a chosen setting has not
   converged, it is extended, and the extension is reported.
4. **Choosing a setting of each type.** Among settings that pass every gate, the
   lowest L0. Ties go to TopK when choosing one primary type.
5. **Gate revision.** A gate can only be revised by pointing to a measured
   property of this data, and each revision is reported beside its original.

## Smoke-test results (seed 1 unless noted; these are not gate measurements)

Fixed before the grid, using only corpus tokens and the 65,536-token monitor
sample.

**Stop and restart.** A real run on the GPU was force-killed twice (taskkill /F).
After resuming, all weights were bit-identical to an uninterrupted run.

**Starting decoder length.** At lambda 5, 6,000 steps:

| Variant | FVU | L0 | Never fired |
|---|---|---|---|
| Rows start at length 1 | 0.458 | 54.8 | 4,016 |
| Length 1 + L1 warm-up 5% | 0.454 | 56.4 | 3,928 |
| Rows start at length 0.1 | 0.401 | 85.6 | 0 |
| Length 0.1 + L1 warm-up 5% | 0.399 | 85.4 | 0 |

Length 0.1 is adopted. L1 warm-up adds nothing beyond it and is not used.

**Lambda** at length 0.1, 6,000 steps:

| Lambda | L0 | FVU | Never fired |
|---|---|---|---|
| 2 | 288 | 0.238 | 0 |
| 3 | 172 | 0.301 | 0 |
| 5 | 86 | 0.401 | 0 |
| 7 | 53 | 0.483 | 0 |
| 10 | 33 | 0.576 | 0 |
| 15 | 21 | 0.682 | 9 |

Lambda 3, 5, 7 and 15 span L0 about 20 to 200 as rule 1 asks. Rule 1 also said
the scan looks only at L0. FVU was visible in the same output, and every setting
in that span was far above the 0.20 gate at this length, so lambda 1.5 was added
under rule 1's "extending the range". The 4 planned values are unchanged. This is
a deviation from the wording of rule 1 and is reported as one. No label was
involved. TopK k = 64 at 6,000 steps gave FVU 0.286.

**Precision (rule 2).** Lambda 5, 6,000 steps:

| Run | FVU | L0 | Never fired |
|---|---|---|---|
| TF32, seed 1 | 0.40128 | 85.628 | 0 |
| TF32, seed 2 | 0.40103 | 85.982 | 0 |
| TF32, seed 3 | 0.40059 | 85.497 | 0 |
| fp32, seed 1 | 0.40136 | 85.628 | 0 |

fp32 vs TF32 on seed 1 differs by 0.00008 in FVU and 0.0003 in L0, against seed
spreads of 0.00069 and 0.485. Never-fired counts are all 0. TF32 is accepted and
takes 155 s against fp32's 272 s.

## Measurements (`eval_sae.py`)

| Measurement | Definition | Starting gate (sec. 6.4) |
|---|---|---|
| FVU | Sum of squared error / sum of squared normalised input, all held-out tokens; also by class | <= 0.20 |
| Loss recovered | (L_ablated - L_sae) / (L_ablated - L_original); 15% of real tokens masked, same masks, 1,000 held-out windows, seed 20261106 | >= 0.90 |
| Inactive | Features that never fire on any held-out token | <= 0.20 |
| Always-on | Features firing on > 50% of held-out tokens | Flagged above 0.01 |
| Cross-seed | Each feature's highest decoder cosine in another seed, over all ordered seed pairs; matched = >= 0.70 | Mean >= 0.70 |
| Frequency | Full distribution per run | Reported |

Loss recovered runs on all 3 seeds of the best setting of each type.

## Deferred

- Logit agreement and comparison with PCA, ICA and raw neurons (sec. 6.4): D7.
- Positive-control thresholds (sec. 6.7): those controls use evaluation-set
  promoters, whose activations are not extracted. This needs its own decision.

## Amendments to the proposal recorded here

- Windows are 1,002 bp, not 1 kb (recorded in layer selection).
- Promoter count is 44,137 after the N rule, against 45,119 in sec. 6.4.
- TopK also uses 3 seeds in the pilot; sec. 6.4 only names seeds for ReLU/L1.
