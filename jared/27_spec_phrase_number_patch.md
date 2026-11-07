# Spec: what the horizon phrase carries, by donor patch (draft 2026-11-07, not approved)

Status: **draft, waiting for approval.** Qwen3-8B, thinking on, one 4090/5090 box. Follows `22` §5 and `01` §5 item 4,
and the scan result in `26` (§2: no cell above its random twin, and the manifold step moved the written horizon in
1 of 130 traces where E14's joint steer moved 47 of 128).

## 1. Question

The donor patch at the phrase works (E14: 104/104 traces write a target-regime horizon, +9.9 log-odds) and the
manifold step mostly does not (+1.4). E10 showed the last token (the unit word) carries the unit and not the number.
Open: **where is the number carried, and is the manifold step the wrong direction, the wrong place, or too small?**

## 2. Design

**Pairs.** Same-unit crossing pairs: a recipient prompt and the same prompt rendered with another horizon in the
*same unit* (e.g. 12 years -> 80 years) that crosses one of the prompt's delays, so the regime changes, with the same
phrase token length (Qwen tokenises digits singly, so lengths must match for a position-wise patch). 24 pairs, balanced
over none->near and both->near, built with E14's donor search restricted to same-unit donors. Donor states come from
a forward pass of the donor-rendered prompt, as in E14.

**Conditions** (per pair, thinking on, patch in both prefill passes, layer 16 unless stated):

| # | condition | tests |
|---|---|---|
| 0 | unpatched rerun (different batch order) | noise floor |
| 1 | self patch (recipient states into recipient) | plumbing, must change 0 labels |
| 2 | full-phrase donor patch | ceiling (E14: ~99% write the donor horizon) |
| 3 | number-token donor patch | is the number carried at the number tokens |
| 4 | unit-token donor patch | must be null on same-unit pairs (E10: 0/6) |
| 5 | top-8 subspace donor patch, number tokens | is the horizon subspace enough |
| 6 | manifold step, number tokens only | the scan's cell, positive control |
| 7 | manifold step, number + unit tokens | E14's steer, positive control |
| 8 | random, norm-matched, number tokens | the scan's control |
| 9 | random inside the 8-dim span, norm-matched, number tokens | control that shares the step's subspace |
| 10 | number-token donor patch at L12 and at L22 | which depth carries the number |

Condition 3 at L12/L16/L22 with 24 pairs; all others at L16 only. About 240 generations, ~45 s each at batch 12:
**~3 h**. Cheap fallback if the box is short: drop 9 and 10, ~2 h.

**Measures.**
- Stated horizon (`exp14_analysis.first_stated_horizon`): does the trace write the donor's number (same unit)?
  Also the shift in decades, and whether it lands in the target regime.
- Choice: change in near log-odds against the rerun, natural closes first, signed toward the donor's regime.
- Label flips against the rerun, reported with the noise floor (about 12-20% of labels).

## 3. Criteria (fixed before running)

- C0: self patch changes 0 labels.
- C1 (number is carried at the number tokens): in condition 3, at least 50% of traces write a target-regime horizon
  and random (8) is at most 10%.
- C2 (the manifold step is the wrong tool): condition 6 reaches less than 25% of condition 3's rate of writing the
  donor's number.
- C3 (the unit token is not where the number lives): condition 4 changes the stated number in at most 1 of the pairs.

Readings: C1 passes and C2 passes -> the number sits at the number tokens in a non-manifold direction; next, fit a
better direction from donor differences. C1 fails -> the number is read from elsewhere (other layers, later tokens);
use condition 10 to localise before anything else. Neither is a claim about how the model computes the choice.

## 4. Cost and stopping

~3 h at $0.39-0.50/h, ~$1.5 with provisioning. Modular as in `26` §6: one condition per unit (a unit is saved when it
finishes), order 1, 2, 3, 6, 7, 8, 4, 5, 9, 10, so the controls arrive first and a stopped run is usable. Sync after
every few units, because this host stopped itself twice last time.

## 5. Open questions for Jared

1. Budget for tonight, and a hard stop time.
2. 24 pairs is a screen. Do you want 48 pairs (twice the cost) instead of conditions 9 and 10?
3. Include the choice4 bank for a half-switch version of the same patch, or keep this on the 2-option prompts?

## 6. Housekeeping

Reuse `exp14_phrase_steer.py`'s donor machinery and `exp15_scan.py`'s resumable unit loop; new code is the same-unit
pair selection, the per-position donor patch and the in-span random. Tests first, as before. `just push` now excludes
dotfiles; set the box label and check `just status` before renting; watchdog limit is 60 minutes.

## 7. Decisions (Jared, 2026-11-07, overnight)

1. 24 pairs (a screen). The overnight budget is a shared $10 hard cap across three experiments, so this one
   must fit about 3 h of one RTX 4090.
2. Conditions 9 and 10 stay in, and they run last, so a short box loses them first.
3. No choice4 variant: this stays on the 2-option prompts.

Implementation notes (code: `analysis/patch_prepare.py`, `experiments/exp16_phrase_patch.py`,
`analysis/patch_summary.py`, `queue_patch.sh`):

- Pairs are drawn from the steerable none-start and both-start prompts with Alan's delays, 12 per transition
  (none -> near, both -> near), seed 0, stratified by bank. The donor is the same prompt rendered with the
  horizon of the same unit word, in the near regime for its delays, with the same phrase token count, nearest the
  log-midpoint of the delays. Only 19 none-start and 35 both-start eligible prompts have one, because the
  grid has 48 horizons and digits are single tokens; donors therefore repeat (e.g. 8 weeks, 11 years).
- Donor states are computed on the box from a forward pass of the donor-rendered prompt (E14's way), not stored.
- The manifold step goes from the prompt's horizon to the donor's horizon (not to the delay midpoint, as E14), so
  that it covers the same move as the donor patch. Condition 6 adds it to each number token, condition 7 adds the
  unit step on the unit token too. Both come from the pooled L16 fit of E14's phrase states. The top-8 subspace of
  condition 5 and the in-span random of condition 9 are that number fit's basis.
- "The rerun" of §2: there are two unpatched units. The baseline runs in the inputs' order and is the reference of
  every patched unit; the rerun runs in another order and is the noise floor (its flips against the baseline).
- Condition 10 is two units, the number-token donor patch at L12 and at L22.
- 13 units of 24 generations, 312 in all, at batch 12 (two batches per unit).
