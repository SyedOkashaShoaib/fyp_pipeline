# Layer selection protocol (frozen 19 Oct 2026, before any probe was run)

This note fixes how the layer is chosen and how the drought-label comparison
is run. It is committed before `probe_layers.py` or `drought_probe.py` has
produced any result, and is not changed after results are seen.

## Inputs

- Probe windows: `data/probe_windows.fa`, 3,998 corpus promoters from
  `data/probe_sample_L1002.tsv`, 1,002 bp, content SHA-256 prefix
  `ba293927e40e3316`.
- Layer summaries: `data/probe/layer_summaries.npz` (not in git), from
  `layer_sample.py --layers 0..40 --dtype fp32 --batch 32`; 167 real tokens per
  window; SHA-256 `92217c65c288d127…`. Layer 40 is `hidden_states[40]`
  (after the final layer norm).
- Motif targets: best log-odds score per window for each of the 309
  selection-set motifs in
  `data/motifs/official_partition_candidate/motifs_selection.meme`
  (18 JASPAR clusters), from `scan_motifs.py`.

## Primary rule: motif probe (label-blind)

1. Ridge regression from the 1,500 standardised dimensions of each layer to
   the standardised motif scores; penalty chosen inside each training fold.
   5 folds grouped by homoeologue/paralogue `group_id`, identical for every
   layer.
2. Score per fold = held-out R² per motif, averaged within each motif cluster
   (`group` in `motif_partition.tsv`), then averaged across clusters.
   Primary summary: **max**-pooled activations. Mean pooling is reported
   alongside but never decides.
3. Best layer = highest mean score among layers 1–40.
4. A layer is tied with the best if the mean of its per-fold difference from
   the best is within one standard error of that paired difference
   (SE = sd of the 5 differences / √5).
5. Choice = the tied layer nearest mid-stack (layer 20); if two are equally
   near, the lower one.
6. Layer 0 (embedding, no context) cannot be chosen. If it falls within the
   tie set, that is reported prominently.
7. Hedge: a runner-up outside the tie set but within 2 paired SE is reported
   as a candidate for a second extraction.

## Secondary comparison: drought-label probe (amendment, option B)

Run as a noisy comparison for the report. It **cannot change** the layer
chosen above.

1. Genes: **development split only** (frozen
   `splits(20260829) 03-09-2026.tsv`); discovery and confirmation stay sealed.
   Shoot and root separately; control pool N0.
2. Pairs: propensity matching as in `split_and_match.py` (caliper 0.10,
   1:1 without replacement, subgenome exact, within-group pairs forbidden),
   seed `20260831`, shoot first and then root. Pairs written to file.
3. Windows: 1,002 bp, built from `cov_dry.tsv` coordinates by
   `extract_windows.py`. **Any pair whose case or control window contains a
   non-ACGT base is dropped**, after matching. SMD is re-checked after the drop.
4. Features: case − control difference of max-pooled activations per layer
   (mean pooling reported too). Columns are scaled on training folds without
   centring.
5. Model: L2 logistic regression with no intercept (the conditional-logistic
   form). A random half of pairs (seed `20260831`) has its sign and label
   flipped so both classes are present. Penalty chosen inside training folds.
6. Folds: 5, grouped by connected components of the graph that links each
   case's group to its control's group.
7. Metric: held-out paired accuracy (share of pairs where the case scores
   above its control; 0.5 = chance), with a 95% bootstrap CI over components.
8. Baselines: layer 0 and a covariates-only model on the covariate
   differences.
9. Root is primary; shoot is reported alongside. Layers are compared by
   paired per-fold differences.
10. Interpretation, fixed in advance:
    - flat within noise: labels give no guidance and the motif choice stands;
    - peak inside the motif tie set: convergent evidence;
    - peak outside the motif tie set and beating the motif-chosen layer by more
      than 2 paired SE in root: that layer is a candidate for a secondary SAE
      in FYP 2. The primary layer does not change.

Seeing development-split label–activation relations is recorded in D4 and the
amendments log; any later development-split decision is made knowing them.
