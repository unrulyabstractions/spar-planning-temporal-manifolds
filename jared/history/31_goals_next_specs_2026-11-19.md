# Goals to draft later (2026-11-08)

A list of what to spec next, not a spec. Each item says what it settles and where it comes from. Nothing here is approved
or costed; cost is guessed only where an earlier run gives a basis.

## A. Steering and patching at the horizon phrase (follows `30`)

1. **Linear steer at the number tokens.** One direction (mean of donor-minus-recipient number states over many same-unit
   pairs), added at the number tokens, same length and random controls as the curve step, at L12, L16, L22. Settles whether
   the curve's geometry adds anything at this position; earlier straight-line arms (E8, `08`/`13`) were at the answer slot,
   and E14, the scan and `27` have none at the phrase. Needs held-out pairs, so the direction is not fitted on the test pairs.
2. **A direction fitted from donor differences.** The donor patch hits 24/24 and the pooled curve step 3/24 (states the
   donor's number). Fit the direction from the real differences at the number tokens and test it the same way. Combine
   with item 1.
3. **Layer sweep of the number-token donor patch.** Only L12, L16, L22 were run, all 24/24. Run the layers between and
   below (and the first layers), to find where it stops working. Cheap: one unit per layer, about 3 min each at batch 12
   by the `27` run time (24 generations per unit).
4. **Steer at more than layer 16.** The `27` conditions 6-9 ran at L16 only. E14 steered at L14, L16, L18 with similar
   effects, so first ask whether it is needed.
5. **What carries the choice, as opposed to the written horizon.** The patch moves the stated horizon and the choice follows,
   as in E14. Not shown: whether the choice is computed from the stated horizon. Candidate: patch after the stated horizon is
   written, or hold the written horizon fixed and patch the phrase.

## B. The manifold fit (follows `30` addendum)

6. **Replot with raw per-horizon means beside the fitted curve.** The unit adjustment creates the PC1 slope; the figures
   show the adjusted line. Show both, and say which one a step uses.
7. **PC5's one-decade period.** Label PC5 by the number's leading digit; refit on random halves of the prompts and see whether
   PC2-8 reproduce. Settles whether the zigzag is digit identity, noise, or something else. Also fit per unit word.
8. **A fit that does not remove the unit.** The unit factor is nested in the horizon. Options: fit within each unit and
   compare, or fit only on the durations written under several units. Decide what "the horizon direction" should mean
   before refitting.
9. **Does the unit adjustment affect the scan (`26`)?** Its cross-unit targets use the absolute curve, so they inherit the
   confound. Check how many of the 40 cells were cross-unit.

## C. Side experiments (follows `28`, `29`)

10. **Implied horizons: the reference check failed (C0).** Explicit-twin accuracy 0.875 (near) and 0.94 (both). Decide whether
    to rewrite the twins or relax the check; until then the family results are provisional. Possibly drop M3 (89% forced
    closes), as decision 4 intended.
11. **Think budget: loops.** 20/57 and 11/35 looping traces closed, and +8,192 closed none of the rest. Next: detect loops early
    and stop them, or sample again instead of extending.
12. **Patch on choice4/8.** Same-unit pairs for the half-switch version of `27`; excluded from `27` (decision 3).

## D. Housekeeping

13. **Commit, push to `jared`, open and merge** (the commit `2a2eebe` exists; push and merge not done). The figure script,
    its figures, the addendum and this file are uncommitted.
14. **Check spend against vast.ai billing.** `just burn` is hours times rate and ignores bandwidth; the key lacks billing read
    permission (`show invoices` returned 401).
15. **Watchdog text and log.** `services.sh` still says "kernel idleness"; `watchdog.log` is truncated on resume; `CLAUDE.md`'s
    description of the activity check should match the code (the watchdog does not detect a finished queue).
16. **A box-side finish marker that stops the box.** Nothing watched the box between my turns; the hourly check was added late.
    A queue that touches `night.done` could also trigger a stop after a sync.
