# Spec: a diagnostics report at the end of every steering run (draft 2026-11-26, not approved)

Branch `qwen8b-steering`; code goes to `evaluation/` as a PR for Augusto. Decided 2026-11-26 (Jared): diagnostics
**report, never gate**; everything in one report that can be looked at. Defaults taken where Jared did not answer: T0
bank first (eval-type banks later), the fp32 Qwen3-8B run (`SteeringQwen8B.md`) on hold until this exists and serves as
its first real test, no checker API spend.

## 1. Why

The Qwen3-8B run of 2026-11-25 spent ~3 GPU-hours on a CAA direction whose problems were visible on CPU beforehand,
partly in files it had already written (`fit.json` recorded cos_choice 0.45 and the 53-vs-346 class mix). Nothing put
those numbers in front of a reader. Found afterwards from saved artifacts (`SteeringQwen8B.md` §1): CAA's classes
barely differ in behaviour (far rate 0.47 vs 0.31); `caa-within` is noise (split-half cos <= 0.2) and mostly the answer
letter at `label` (cos 0.67); calibration froze x = 0.8, past the dose peak at 0.2, where the effect has reversed.

## 2. What it is

`evaluation/diagnose.py <run dir> [--artifacts DIR --bank DIR --baseline DIR]` writes into the run directory:
- `diagnostics.html`: one self-contained page (inline CSS and SVG, no network, no new dependencies), opened from disk;
- `diagnostics.json`: every number on the page;
- `diagnostics_items.csv`: the per-prompt table (§3.A).

It reads only files that runs already write (bank `prompts.jsonl` and `checks/`, baseline `responses.jsonl`,
activations `index.jsonl` and `L*_<site>.npy`, method bundles `method.json`/`arrays.npz`/`fit_uids.jsonl`/`fit.json`,
`calibration.json`, `report.json`, `n0_check.json`). A section whose inputs are missing says so and is skipped, so it
runs right after fitting (before any GPU replay) and again at the end. `run_local.sh` calls it as its last step;
CPU only, seconds.

Every check shows its value, a threshold and a flag (ok / look), with one line on what a "look" means. Thresholds are
constants at the top of the file, chosen from the 2026-11-25 numbers, to be revised.

## 3. Sections

**A. Data and labels.** Counts by split, regime, label style, option order, phrasing, unit rewrite. Checker agreement
and problem flags by category (T0: 700/700 agree; 460 flags). The per-prompt table: uid, split, fold, regime, correct
option, H/far delay, label style, far label, near first, model's choice, correct?, hit the token cap, trace length,
CAA class, within-regime class.

**B. Baseline behaviour.** Accuracy by regime; far rate on `none` prompts (flag: answered like a long horizon);
P(far) against log10 H/D on the verification grid (plot); letter bias (P(answer a) when far is a vs b) and order bias.

**C. Each fitted direction** (every bundle in the artifacts, at its site):
- class sizes by regime, and the far-rate gap between its + and - classes (flag: gap < 0.3);
- split-half reliability: median cosine of the direction refitted on two random halves of its fit prompts, 30 splits
  (flag: < 0.7);
- norm over typical norm;
- cosine table with the nuisance directions (answer letter, label style, option order, hit the cap) and with
  `choice`, regime (`both` minus `near_only`) and reward (flag: |cos| > 0.3 with a nuisance);
- the same reliability and nuisance cosines at every other extracted site, side by side, to show where a direction
  is cleaner.

**D. Strength.** The calibration dose curve (odd effect and label mass against x, plot), the x the rule chose, the
peak; flag when the chosen x is past the peak or the effect changes sign along the grid; held-out vs verification-grid
sign when `report.json` exists.

**E. Outcomes** (when `report.json` exists). Per arm: meaning, letter, even part, p against each null family,
switch-point shift, plateaus and slope, as `analyze.py` computes them. Flags: letter larger than meaning; `choice` (the
positive control) not run, or not beating the nulls ("the readout cannot show an effect; read nothing else").

**F. Readout noise.** N0 (hook-free and batched vs one-at-a-time), replay dtype; flag: batched-vs-alone difference
larger than half the largest meaning effect.

At the top: a summary of every "look" flag with a link to its section.

## 4. Tests (`evaluation/test_diagnose.py`, synthetic inputs, CPU)

- A direction built from the letter plus noise is flagged on letter alignment; a pure-noise direction on reliability.
- A class split with equal far rates is flagged on the gap.
- A dose curve that peaks and reverses, with the rule's x past the peak, is flagged.
- Missing optional inputs give a page with those sections marked skipped, not an error.
- The HTML is well formed and holds no external URLs.

Acceptance on real data: run on `~/SPAR/results/qwen8b_quick_20261125/`; the page must show the numbers of §1 (CAA
split-half 0.35 at `label`, gap 0.47 vs 0.31, `caa-within` letter cos 0.67, chosen x past the peak) flagged.

## 5. Out of scope (later specs)

Gating; confound checks C3-C8; a planted-signal test of the full statistics path; the checker on the eval-type banks
(API cost); minimal-pair fitting data; eval-type banks in the report.

## 6. Tasks

1. Per-prompt table and sections A-B, with tests.
2. Section C (directions, reliability, nuisance cosines, all sites), with tests.
3. Sections D-F, the summary and the HTML page, with tests.
4. `run_local.sh` hook; acceptance run on the 2026-11-25 artifacts; look at the page; PR to `evaluation/`.
