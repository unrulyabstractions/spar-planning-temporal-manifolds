# Results: horizon and decision neurons against Neuron Lens (stage A 2026-11-12, stage B 2026-11-13)

Spec `32_spec_neuron_lens.md`. Qwen3-8B, thinking on, full bank, saved E0 activations;
labels from the Neuron Lens catalog (`stdlogits/neuron-lens` @ `21666b9`, CC BY 4.0, by
a9lim and treetown with Claude), weights from `Qwen/Qwen3-8B` @ `b968826`.
Outputs: `$RESULTS/analysis/neuron_lens_compare.{json,csv,png}` (top 20 per list, with
labels and explorer URLs). Code: `analysis/neuron_lens_fetch.py`, `analysis/neuron_lens_compare.py`.

## Summary

1. **No sparse set of horizon neurons.** The horizon is carried by most of the layer. At
   the suffix newline (L16), 9,860 of 12,288 neurons pass the family-wise permutation
   threshold for raw log-horizon (|rho| > 0.20), and 6,269 still pass within unit. At
   the pre-label token, 211 (L22) and 235 (L24) pass within unit. A top-20 list from
   thousands of passing neurons is not special, and its labels are unrelated to time
   (theatre, personality nouns, statistics phrasing at L16).
2. **The frozen keyword test finds nothing that survives correction.** Across 18 lists × 2
   families (36 Fisher tests), the smallest p values are 0.003 (time, L21 writes vs the
   L22 raw-horizon direction) and 0.006 / 0.011 (choice, L24 raw horizon, A1 / A2).
   Bonferroni over 36 tests puts the smallest at 0.11. Within-unit lists: nothing.
3. **One suggestive write-side cluster (L21 → L22, raw horizon).** Among the top 8 writes
   aligned with the pre-label horizon direction, 4 are labelled as long-duration neurons:
   n8408+ "long duration or distance → long-term time spans (yearly, decades, centuries)"
   (conf 0.7), n10101+ "months and seasons → year-count words" (0.6), n11315- "years,
   during, until" (0.5), n9501+ "→ lifelong, long-term, decades". Their cosines (0.06-0.08)
   sit at the random-direction threshold (0.072), so the alignment is weak. This is the raw
   horizon, so it can be the unit word.
4. **Post hoc, not tested: the decision lists read as magnitude and comparison.** At L22
   (A1) the top decision neurons are labelled "decrease and reduction", "small, low,
   less", "after, later, larger", "initial/start", "storing for later/future use → future,
   forever", "alternate timelines". At L21 (A2) the writes aligned with the decision
   direction are "larger, longer, more → fewer, smaller, lower", "minimal, lowest",
   "decrease". This fits a near-vs-far choice read as small/soon vs large/later. The
   frozen families have no magnitude words, so this was not tested; testing it now would
   need a new family fixed before looking, on a held-out cell.

## Table (A1 read: approximate activations; A2 write: block L-1 writes vs ridge direction)

| cell | target | A1 n | A1 thr | A1 pass | A2 thr | A2 pass | time hits A1 / A2 | choice hits A1 / A2 |
|---|---|---|---|---|---|---|---|---|
| suffix newline L16 | horizon raw | 471 | 0.20 | 9,860 | 0.072 | 1 | 1 / 0 | 1 / 0 |
| suffix newline L16 | horizon within unit | 471 | 0.21 | 6,269 | 0.071 | 3 | 2 / 2 | 0 / 0 |
| suffix newline L16 | decision | 471 | 0.20 | 5,833 | 0.072 | 1 | 0 / 2 | 0 / 0 |
| pre-label L22 | horizon raw | 371 | 0.22 | 5,828 | 0.072 | 2 | 0 / **6** (p 0.003) | 0 / 0 |
| pre-label L22 | horizon within unit | 371 | 0.23 | 211 | 0.071 | 1 | 3 / 0 | 0 / 0 |
| pre-label L22 | decision | 371 | 0.23 | 7,374 | 0.070 | 11 | 4 (p 0.06) / 1 | 0 / 0 |
| pre-label L24 | horizon raw | 371 | 0.22 | 5,122 | 0.071 | 3 | 2 / 3 | 3 (p 0.006) / 3 (p 0.011) |
| pre-label L24 | horizon within unit | 371 | 0.23 | 235 | 0.073 | 2 | 0 / 2 | 0 / 0 |
| pre-label L24 | decision | 371 | 0.22 | 6,944 | 0.070 | 3 | 2 / 2 | 0 / 0 |

Hits are out of the top 20. Base rates over all poles of the layer: time 5-8%, choice 2-3%.

## Caveats

- **A1 is approximate.** It skips block L's attention output, which was not saved. Stage B
  measured the error: per-neuron r 0.73-0.85, and A1's top-20 lists mostly don't hold up.
  Read the stage A lists above through stage B.
- **Spec deviation (pole assignment).** The spec said to rank poles by signed rho. A
  neuron whose activation stays positive and fades as the horizon grows would then be
  reported as its negative pole, whose label describes contexts where it never fires.
  So each neuron is ranked by |rho|, and the pole reported is the sign of its median
  activation at the cell. A2 reports the pole whose write points toward a larger target.
- The labels describe neurons on web, code and chat text (15M tokens), not on our
  prompts. 31% of L21-L27 firing labels reach AUROC 0.8 in the catalog's own check, so
  labels at these depths are less reliable than early ones.
- Keyword matching is coarse. "Long", "later" and magnitude words partly fall outside
  the frozen families.

## Stage B (exact activations, 2026-11-13)

One forward pass per prompt on a 4090 (`experiments/neuron_acts.py run`, about 2 min),
hooking the real MLP neuron activations at blocks 16, 22 and 24. Run on 459 of the 471
prompts: 12 were dropped because their cached trace differs from E0's (n = 362 at the
pre-label token). Outputs: `$RESULTS/analysis/neuron_lens_exact.{json,csv,png}`. The
"A1" columns below are stage A rerun on the same 459 prompts, so A1 and B are paired.
The saved E0 activations match the box block input (median cos 1.000), so the sequences
are E0's.

1. **A1's error is large at the neuron level, and the top lists are mostly artefacts of
   it.** Median per-neuron r(approx, exact) is 0.73 at the suffix newline (L16), 0.82 at
   L22 and 0.85 at L24. Over neurons, r(rho approx, rho exact) is 0.65-0.85. Top-20
   overlap between A1 and B is only 1-7 of 20. The error is largest at L16, not at the
   pre-label token as the stage A caveat expected. Stage A's density finding stands:
   9,586 (raw horizon) and 6,100 (within unit) L16 neurons pass with exact activations,
   and 257 (L22) and 141 (L24) pass within unit at the pre-label token.
2. **The L22 decision list keeps its magnitude theme, and the core neurons survive.**
   n6617- "after, later, larger comparatives", n11774+ "decrease and reduction",
   n6444+ "small, low, less", n11324+ "comparison words ('than', 'compared to')" and
   n1630+ "storing things for later/future use" are in both the A1 and the B top 20.
   B adds n5934- "'Finally' after a long wait → eventually, at last" and n12117+
   "frequency words: periodically, monthly". The L24 decision list does not have this
   theme (1-2 magnitude labels out of 20, e.g. n7101- "larger, higher, bigger"). This is
   still post hoc: no frozen family covers magnitude words.
3. **No frozen keyword enrichment survives correction with exact activations.** Over the
   9 B lists × 2 families (18 Fisher tests), the smallest p is 0.003 (time, L22 decision,
   6/20 against a 7.7% base rate), which is 0.056 after Bonferroni. Its hits are the
   "later / finally / periodically / future" neurons from point 2, so it is the same
   observation, not independent support. Stage A's L24 raw-horizon choice hit
   (3/20, p 0.006) drops to 1/20 (p 0.31), so it came from the approximation.

| cell | target | A1 pass | B pass | per-neuron r | r(rho) | top-20 overlap | B time hits (p) | B choice hits (p) |
|---|---|---|---|---|---|---|---|---|
| suffix newline L16 | horizon raw | 9,895 | 9,586 | 0.73 | 0.69 | 2 | 3 (0.19) | 1 (0.35) |
| suffix newline L16 | horizon within unit | 6,333 | 6,100 | 0.73 | 0.69 | 7 | 1 (0.80) | 0 (1) |
| suffix newline L16 | decision | 5,819 | 5,445 | 0.73 | 0.65 | 1 | 2 (0.46) | 1 (0.35) |
| pre-label L22 | horizon raw | 5,661 | 5,007 | 0.82 | 0.80 | 6 | 2 (0.46) | 0 (1) |
| pre-label L22 | horizon within unit | 203 | 257 | 0.82 | 0.78 | 5 | 3 (0.20) | 2 (0.11) |
| pre-label L22 | decision | 7,250 | 6,945 | 0.82 | 0.80 | 6 | **6 (0.003)** | 0 (1) |
| pre-label L24 | horizon raw | 4,704 | 4,470 | 0.85 | 0.84 | 5 | 1 (0.67) | 1 (0.31) |
| pre-label L24 | horizon within unit | 251 | 141 | 0.85 | 0.84 | 6 | 1 (0.67) | 0 (1) |
| pre-label L24 | decision | 6,887 | 6,393 | 0.85 | 0.85 | 6 | 1 (0.67) | 0 (1) |

Per-neuron r is the median over neurons of r(approx, exact) across prompts. It is the same
for every target in a cell because it does not depend on the target.

Cost: $0.27 total, of which $0.20 went to two rentals of one host (machine 155376, now in
`EXCLUDE_MACHINES`) that never finished loading.

**Follow-up (not specced or approved).** The magnitude reading at L22 is now the most
stable thing in this analysis. Two ways to test it: freeze a magnitude keyword family and
test it on a held-out cell, or (sharper) steer the L22 magnitude neurons and the L21
long-duration writes (n8408+, n10101+) at the pre-label token and measure near log-odds,
as in E8.
