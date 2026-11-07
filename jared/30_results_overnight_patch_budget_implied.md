# Overnight 2026-11-07: phrase patch (27), think-budget extension (28), implied horizons (29)

One RTX 4090 box (59546743, $0.421/h), tmux queues, 10:06 to ~20:00 UTC. Spend $6.33 by `just burn`
(estimate, not a bill: it is hours x rate and ignores bandwidth; billing API not readable with this key).
Box destroyed after a full sync. All three are screens (24 pairs / 105+74 traces / 288 rows), no significance claims.
Raw summaries: `results/horizon/exp16_*/patch_summary.txt`, `exp17_*/think_budget_summary.txt`, `exp18_*/implied_summary.json`.

## 27: where the number lives (phrase patch, 24 same-unit crossing pairs, L16)

| condition | states donor's number | lands in target regime | d near log-odds (natural) |
|---|---|---|---|
| self patch | 0/24 | 0/24 | -0.02 |
| full phrase donor | 24/24 | 24/24 | +5.74 |
| number-token donor | 24/24 | 24/24 | +5.16 |
| unit-token donor | 0/24 | 0/24 | +0.16 |
| top-8 subspace donor | 12/24 | 20/24 | +0.72 |
| manifold step, number tokens | 3/24 | 8/24 | +6.18 (n small, see below) |
| manifold step, number+unit | 4/24 | 9/24 | +2.10 |
| random, number tokens | 0/24 | 0/24 | -0.61 |
| random inside the 8-dim span | 0/24 | 0/24 | +1.03 |
| number donor at L12 / L22 | 24/24 / 24/24 | 24/24 / 24/24 | +4.33 / +5.92 |

Criteria C0-C3 all pass. Reading: the number is carried at the number tokens, at L12, L16 and L22 alike (so depth does not
localise it), and not at the unit token. The fitted manifold step is the weak tool: it reaches the donor's number in 3/24
against 24/24 for the real states, and the top-8 subspace alone gets half. The in-span random does as little as the isotropic
one, so the step's failure is not just a norm effect.
Caveats: the noise floor is 5/24 label flips (rerun vs baseline); the "natural" log-odds column uses only pairs where
neither trace was cut off, and the manifold-number +6.18 comes from few such pairs and should not be read as an effect
(its forced column differs a lot, 16 forced vs 4-10 elsewhere). Donors repeat (only 19 none-start and 35 both-start prompts
have a same-length donor). The patch writes the donor's number in the trace, so this shows the number is *read* from the
phrase positions, not that the choice is computed there.

## 28: more thinking tokens for force-closed choice traces

| bank | forced traces (looping) | closed within +3072 | wrong->right / right->wrong | closed loop / no-loop |
|---|---|---|---|---|
| choice4 | 105 (57) | 68 (65%) | 13 / 2 | 20/57, 48/48 |
| choice8 | 74 (35) | 46 (62%) | 7 / 0 | 11/35, 35/39 |

P0 (rerun reproduces the original label) passes: 103/105, 73/74. C2 (late-position bias disappears after closing) passes.
C1 fails narrowly on accuracy (choice4 exact 84% vs 85% needed; choice8 78%) and choice8 net fixes (7 vs 7.05 needed).
C3 fails: looping traces rarely close (20/57, 11/35). Extending to 8,192 closed none of the 37 and 28 that stayed open
(25 and 27 attempted). So the forced-close errors are partly a budget problem (the closed traces become accurate and lose the
late-slot pick bias: choice4 excess-late +0.33 -> +0.03), and partly loops that more tokens do not fix.

## 29: implied horizons (288 rows: M1 144, M2 72, M3 48, M4 24; readout probes trained on E0 think-on activations)

- C0 (reference valid) **fails**: explicit-twin accuracy 0.875 (near), 0.94 (both), under the threshold. Treat everything
  below as provisional.
- Behaviour, implied vs explicit twin (accuracy, near/both): A 62/88 vs 81/94, B 56/88 vs 94/88, C 94/81 vs 88/100.
  Only family C tracks (C1 pass for C, fail for A and B). Paired McNemar p = 0.34, 0.11, 0.69: the implied-vs-explicit gap
  is not significant at this n.
- Conversion (what horizon the trace writes): A 40/40 and C 40/40 state a duration (median error 0.04 and 0.001 decades);
  B states one in 35/40 and 69% fall inside the target interval. Misconversions that flip the regime: A 8, B 5, C 0.
- Mediation: of 7 regime-flipping misconversions, the choice follows the stated regime in 1 (14%). C4 underpowered.
- Null twins pick near 12% / 25% / 50% (A/B/C), so the no-horizon default is not clean either.
- Readout (probe MAE in decades): implied is worse than explicit at the suffix (e.g. B 0.48 vs 0.23, A 0.90 vs 0.39) but
  rank-correlation stays high (rho 0.7-0.9). At think@1 the implied horizon is weakly present (B rho 0.24-0.53).
  C3 verdict per cell is in the printout; I have not turned it into a single claim.
- M3 (none regime) ran although decision 4 dropped it from the criteria; 89% of its traces were force-closed.

## Process notes

- The host connection dropped during two syncs (rsync code 12/255); the box itself and tmux survived. Syncs resume,
  so I reran them. No host stop this time.
- Nothing monitored the box between my turns until I scheduled an hourly check partway through. The idle watchdog would
  not have stopped a finished queue.

## Open

- Promote C0 failure of 29 into a decision: rewrite the explicit twins or relax the reference check?
- Better direction for 27: fit from donor differences at the number tokens (the donor patch works, the pooled manifold does not).
- Uncommitted: all of 27/28/29 code, tests and this note; `infra/vast.sh` duration filter.
