# SAE pilot results (deliverable D6, draft)

Working record. The formal version for supervisor review is "Label-Blind Sparse
Autoencoder Pilot on AgroNT Layer 13": https://claude.ai/artifact/7cyV1ZhLpWUPriJsfp911X

Run 2026-11-04 to 2026-11-06. Settings are in `docs/sae_pilot_protocol.md`. All
numbers come from `sae_pilot/summary.tsv`, `crossseed.json`, `loss_recovered.json`
and `loss_recovered_unmasked.json`. No response label was read at any point.

**Status: gates are not frozen.** No setting passes every gate as the protocol
defined them, so the choice of setting and gates needs a decision (see the end).

## Result

Every setting fails at least one gate as defined. The inactive and always-on
gates pass everywhere. FVU, loss recovered and cross-seed matching pull against
each other: settings that rebuild layer 13 better have less stable features.

Means of 3 seeds, on 1,728,283 held-out tokens from 10,349 windows. Bold = passes.

| Setting | L0 | FVU (<= 0.20) | Loss recovered (>= 0.90) | Mean max cosine (>= 0.70) | Matched, pairwise | Stable in all 3 seeds | Inactive (<= 0.20) | Always-on (<= 0.01) |
|---|---|---|---|---|---|---|---|---|
| ReLU, lambda 1.5 | 390 | **0.196** | 0.829 | 0.643 | 43% | 2,081 | **0** | **0** |
| ReLU, lambda 3 | 159 | 0.297 | 0.708 | **0.737** | 58% | 2,931 | **0** | **0** |
| ReLU, lambda 5 | 78 | 0.396 | 0.559 | **0.773** | 63% | 3,266 | **0** | **0** |
| ReLU, lambda 7 | 46 | 0.476 | 0.438 | **0.784** | 66% | 3,440 | **0** | **0** |
| ReLU, lambda 15 | 13 | 0.676 | 0.182 | **0.855** | 95% | 5,456 | **0.0005** | **0** |
| TopK, k = 32 | 32 | 0.345 | 0.783 | 0.643 | 44% | 2,135 | **0.0008** | **0** |
| TopK, k = 64 | 64 | 0.273 | 0.784 | 0.556 | 31% | 1,415 | **0.0001** | **0** |
| TopK, k = 128 | 128 | 0.204 | 0.874 | 0.449 | 20% | 795 | **0** | **0** |

"Stable in all 3 seeds" counts seed 1 features with cosine >= 0.70 to some feature
in both other seeds, out of 6,000.

Seeds agree closely: FVU within 0.001 and L0 within 1.2. Loss recovered varies more,
by up to 0.046.

## What the numbers mean

- **FVU.** Only lambda 1.5 clears 0.20. TopK k = 128 misses by 0.004 and had not
  converged (below).
- **Region classes.** FVU is similar across classes. Repeat windows rebuild best
  and introns worst: 0.176 and 0.204 at lambda 1.5, 0.175 and 0.214 at k = 128.
- **Feature frequency.** Sparsity has a real cost at the setting that passes FVU.
  At lambda 1.5 the median feature fires on 6.9% of tokens, and 390 of 6,000 are
  active per token. At k = 128 the median is 1.0%.
- **ReLU vs TopK.** TopK recovers more masked-LM loss at the same FVU (k = 32:
  FVU 0.345, recovered 0.783; lambda 3: FVU 0.297, recovered 0.708). The L1 penalty
  shrinks activations, which is a likely reason, but I did not test it. ReLU/L1 is
  more stable across seeds at the same L0.

## Convergence (rule 3)

Convergence is measured over the last pass of the constant-rate phase. All 15 ReLU
runs and all TopK k = 32 and 64 runs improved by less than 1%, so they converged.
**TopK k = 128 did not: it improved 1.4-1.6%.** It is reported as not converged and
would need extending if chosen. Five passes is enough for every other setting.

## The masked-token finding (diagnostic, added after seeing results)

The loss-recovered measurement swapped x-hat in at every real token, including the
15% replaced by `<mask>`. The SAE never trained on `<mask>` vectors, because the
store holds unmasked activations. So I measured it again leaving masked positions
untouched. **This was not in the protocol and was chosen after seeing the gate
fail**, so it is a diagnostic, not the gate measurement.

Mean ablation at unmasked tokens only gives loss 8.204 (8.469 before). The original
loss is 7.263 in both.

| Setting | Seeds | Recovered, as defined | Recovered, masked tokens untouched |
|---|---|---|---|
| ReLU, lambda 1.5 | 3 | 0.829 | **0.919** |
| ReLU, lambda 3 | 3 | 0.708 | 0.862 |
| TopK, k = 32 | seed 1 only | 0.783 mean | **0.910** |
| TopK, k = 64 | seed 1 only | 0.784 mean | **0.912** |
| TopK, k = 128 | 3 | 0.874 | **0.943** |

Much of the shortfall comes from rebuilding `<mask>` vectors, not from losing
information at real tokens. The proposal's wording ("when the reconstruction is
substituted") does not settle which version is meant.

## Ambiguity in the cross-seed gate

The proposal says ">= 0.70 mean maximum cosine similarity for a matched feature;
matched fraction reported" (sec. 6.4), and sec. 6.6.3 aggregates "only features
satisfying the predeclared cross-seed matching rule". That can be read two ways:

- **Dictionary gate:** the mean over all features must reach 0.70. The protocol
  wrote it this way, and it is what the table above tests.
- **Per-feature rule:** 0.70 decides which features count as matched, and the
  matched fraction is reported, not gated. Matched features average about 0.85.
  Under this reading no setting fails, and the stable count is what matters.

The power figures in sec. 6.6.5 assume 4,800 features screened. Stable features are
well below that at every setting except lambda 15, which would need recomputing.

## Corpus sufficiency

The dictionary is well used on 103,343 windows (15.53M training tokens): no setting
has more than 0.13% inactive features. There is no sign the corpus is too small, and
the 12,850 reserve promoters are not needed.

## Checks and run notes

- **Stop and restart.** Two hard kills on the GPU resumed bit-identical to a clean
  run. The grid also stopped for real at 11:25 on 2026-11-04 when a session ended,
  and resumed from step 12,166, reproducing the logged values at step 13,000.
- **Speed.** About 8.8 minutes per run and 3.5 hours for 24 runs, against the
  6.6 hours first estimated. TF32 was accepted under rule 2.
- **Storage.** `sae_pilot/` 8.1 GB, activations 49 GB, under the 100 GB limit.

## Deviations from the protocol

1. Lambda 1.5 added beyond the 4 values planned, after FVU was seen in the scan
   (rule 1 said the scan looks only at L0).
2. Loss recovered measured on all 24 runs, not just the best setting of each type,
   because no setting passed the other gates.
3. The masked-token diagnostic above.

## Limits

- Loss recovered uses 1,000 windows and 25,000 masked tokens, and its gap is small
  in absolute terms (7.263 to 8.469).
- FVU is measured on unmasked activations only.
- D7 still has to train and evaluate the frozen setting; these are pilot numbers.

## Still open

- **Gate decision** (below).
- **Feature eligibility threshold** (sec. 6.6.3, set from this pilot): depends on
  the setting chosen.
- **Positive-control thresholds** (sec. 6.7, set from this pilot): those controls
  use evaluation-set promoters, whose activations are not extracted. D6 is not
  complete until this is settled.

## Decision needed

| Option | Setting | What changes | Result |
|---|---|---|---|
| A (recommended) | ReLU, lambda 1.5 | Loss recovered measured at unmasked tokens; cross-seed read as a per-feature rule. No threshold lowered | Passes all gates: FVU 0.196, recovered 0.919, 2,081 stable features. Least sparse option |
| B | TopK, k = 128, extended | Same two readings; needs more training to reach FVU 0.20, not yet shown | Recovered 0.943, only 795 stable features |
| C | ReLU, lambda 3 | Lower FVU to about 0.30; thresholds set after seeing results | Sparser and more stable (L0 159, 2,931 stable), recovered 0.862, which still misses 0.90 |
| D | None | Keep every gate as defined | "No setting passes" (sec. 7.3): features descriptive only, the question passes to the linear probe |

Options A and B rest on readings chosen after seeing results, even though no number
is lowered. That should be stated in the report, and is worth raising with the
supervisor.
