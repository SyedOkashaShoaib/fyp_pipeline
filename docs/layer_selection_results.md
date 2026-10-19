# Layer selection results (19 Oct 2026)

Rule and drought protocol: `docs/layer_selection_protocol.md`, committed before
any probe was run. Figure: `docs/figures/layer_selection_motif_vs_drought.png`.

## Primary: motif probe (label-blind) chooses layer 13

3,998 probe windows (1,002 bp), 309 selection motifs in 18 JASPAR clusters,
41 layers, 5 grouped folds (`data/probe/layer_probe_results.tsv`).

| Layer | Group-averaged R² (max) | Behind layer 13 (paired) |
|---|---|---|
| 0 (embedding) | 0.019 | 0.122 |
| 12 | 0.1376 | 0.0032 ± 0.0002 |
| **13** | **0.1408** | — |
| 14 | 0.1375 | 0.0033 ± 0.0005 |
| 20 | 0.1244 | |
| 40 | 0.0840 | |

- Tie set under the paired rule: {13} only, so the mid-stack tie-break never
  applies. No hedge layer within 2 SE.
- Layer 0 is far below every transformer layer (0.019 against ≥ 0.084): the
  probe is measuring context-dependent content, not 6-mer presence.
- Mean pooling (secondary) peaks at layer 12 (0.178), with 13 just behind.
- The paired SEs are very small (folds move together), so "tie set of one" says
  the peak is reproducible across folds, not that layer 12 or 14 would be a
  materially worse choice.
- Caveats: motif targets are a proxy; JASPAR plant motifs are mostly
  *Arabidopsis*; three 4 bp motifs saturate (e.g. MA2373.1 AZF2, best-score sd
  0.05), giving near-constant targets. Cluster averaging limits their weight.
  The motif partition used is `official_partition_candidate`.

## Secondary: drought-label probe (development split, option B)

Pairs: N0, caliper 0.10, seed 20260831 → shoot 1,353 / root 2,757; 51
development windows had non-ACGT bases, dropping 21 shoot and 38 root pairs →
**shoot 1,332 / root 2,719**. Worst |SMD| after the drop: shoot 0.0995
(log gene length; it was 0.1004 before the drop), root 0.020.
Results: `data/drought/drought_probe_results.tsv`.

| Tissue / pooling | Best layer | Paired acc (95% CI) | Layer 13 | Layer 0 | Covariates only |
|---|---|---|---|---|---|
| **Root / max** (primary) | 7 | 0.535 (0.516–0.554) | 0.524, 0.011 ± 0.011 behind | 0.498 | 0.481 |
| Root / mean | 7 | 0.534 (0.515–0.554) | 0.533, tied | 0.509 | 0.481 |
| Shoot / max | 13 | 0.535 (0.507–0.563) | best | 0.528 | **0.534** |
| Shoot / mean | 19 | 0.540 (0.512–0.568) | 0.535, 0.0045 ± 0.0045 behind | 0.489 | 0.534 |

Interpretation under the pre-declared rules:

- **Root:** the peak (layer 7) lies outside the motif tie set but beats layer 13
  by about 1 paired SE, not the 2 SE needed to trigger a secondary SAE. The root
  tie set {1, 2, 3, 7, 23} is scattered rather than contiguous, which is what
  noise looks like. **The motif choice stands.**
- **Shoot:** layer 13 is the best layer, but a covariates-only model scores the
  same (0.534), so shoot accuracy cannot be separated from residual covariate
  imbalance (log gene length sits at the SMD limit). Treat shoot as
  uninformative.
- The signal is weak everywhere: best accuracies are about 0.535, and the CI
  of a best-of-40 layer is optimistic because of the selection.
- Shape agreement across layers, Spearman ρ between the motif curve and the
  drought curve (layers 1–40): root 0.30 (p = 0.057), shoot 0.61
  (p = 3e-5). Both curves fall off in the deep layers (30–40).

Development-split label–activation relations have now been seen. Record this
in D4 and the amendments log.
