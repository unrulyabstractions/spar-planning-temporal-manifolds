# Qwen3-8B, thinking on: first full evaluation run (2026-11-25)

**Question.** (A) Run from scratch on Qwen3-8B, do the steering methods move the horizon choice on the T0 bank more
than random edits of the same size do? (B) Does the effect carry over to `system_prompt`?

**Short answer.** No. The order-balanced CAA effect is small (+0.22 logits) and no larger than random
covariance-matched directions (p = 0.45). What CAA mostly does at this layer is push one answer letter (+9.8 logits),
and in generation it makes the model reason until the token cap. On `system_prompt`, CAA is again at the null level
(+0.18, p = 0.65), while asking in the prompt moves the answer by +1.2 to +1.7 logits.

## Setup

- First end-to-end run of `evaluation/run_local.sh` on a model other than Qwen3-1.7B: baseline on all 700 T0 prompts
  (thinking on, sampled with Qwen's own T = 0.6, top-p 0.95, top-k 20; cap 4,096 tokens), extraction at layer 24 of
  36 (the decision-direction layer from Jared's E8), label site, fit, the five bundles, hook gate, shared draws.
- Baseline: 663 of 700 prompts usable for extraction (37 skipped: no parsed choice, mostly at the token cap).
  Verification grid: 178 readable of 200. Unsteered replay picks the generated label on 99.4% of items; unsteered
  switch point log10(H/D) = +0.38, slope 3.42; accuracy on the grid 0.71.
- Readout: replay (the model's own reasoning held fixed, answer logits read under the edit), in bf16.
- Strength frozen on held-out extraction prompts (fold 0, 34 prompts): x = 0.8 × typical norm, mean odd effect −1.16
  there with label mass unchanged (0.999).
- Random baseline: 19 covariance-matched directions. Arms: CAA, curve-snap, covariance (the quick set, as on Gemma).
- The full T0 report card (`analyze.py`) and transfer table (`transfer_eval.py`) are not committed; they are
  regenerated from the run directory. The report card's last line ("Plumbing run: the model is small…") is
  boilerplate from `analyze.py` and does not apply.

## Results

| | prompts | meaning (logits) | p vs covariance | switch-point shift (decades), p |
|---|---|---|---|---|
| CAA, T0 verification grid | 178 | +0.22 | 0.45 | +0.65, 0.15 |
| CAA, `system_prompt` | 38 | +0.18 | 0.65 | +0.09, 0.40 |
| "be long-term-minded" appended, `system_prompt` | 38 | +1.68 | – | +0.05 |
| "as if the horizon were 100× longer", `system_prompt` | 38 | +1.17 | – | +0.05 |
| neutral sentence, `system_prompt` | 38 | +0.34 | – | −0.01 |

- **CAA is mostly a letter direction here.** Its letter part is +9.8 logits against +0.22 for meaning, and its even
  part (the same at +x and −x, i.e. disruption) is +3.9. A likely reason, not tested: Qwen's baseline has a position
  bias that differs by choice. On the extraction split, far choices were answered "a)" 58% of the time (47 of 81),
  near choices 41% (65 of 159). The far-minus-near mean at the label token then contains "a) minus b)". This is new
  on Qwen: Gemma's CAA letter part was −0.31 against +1.54 for meaning.
- **The calibration's sign does not hold up.** The held-out fold gave −1.16, the verification grid +0.22 (order-balanced
  in both). A sign flip between folds is what a near-zero meaning effect plus a large letter effect looks like.
- **curve-snap** pushes the wrong way (−0.35), with a large even part (+3.5), and flattens the choice curve (slope
  3.42 → 1.84 / 1.06). Snapping to the curve with no displacement (`curve-snap@0`) moves the meaning by +2.1 logits:
  projecting onto the fitted curve is itself a large edit at this layer. No random baseline was run for curve-snap.
- **Steered generation (16 prompts) breaks.** With CAA at ±x applied to every token, 15/16 (+x) and 11/16 (−x) replies
  hit the 4,096-token cap, against 1/16 unsteered; only 1 and 4 replies parsed. x = 0.8 is too strong for generation,
  even though label mass under replay stays at 0.997.

## Caveats

- One layer, one site, every token edited, reasoning held fixed. K = 19 draws (smallest possible p = 0.05).
- bf16 replay: the N0 check shows up to 0.5 logit between batched (batch 16) and one-at-a-time replay. That is small
  next to the letter effect but comparable to the meaning effect, so a meaning result near the null needs fp32.
- The isotropic and shuffled-label baselines were not run; the covariance family is the demanding one.
- The prompting arms have no null family; their numbers compare to the neutral sentence (+0.34).

## Next

1. Remove the letter component before calling CAA null: refit CAA within answer letter (far−near among "a)" replies
   and among "b)" replies, averaged), or project out the a)−b) direction. The extraction arrays are saved, so this is
   CPU work plus a short replay.
2. Edit earlier (`think_end`, `prompt_end`, both already extracted), where the edit can change the reasoning and the
   answer letter is not yet in the residual stream.
3. For generation, a strength sweep below x = 0.8 (e.g. 0.1, 0.2, 0.4) to find where reasoning still terminates.

Run: one A100 40 GB (Slovenia), about 3.1 GPU-hours including restarts, about $2.50 (plus $0.14 on a first host that
never accepted the SSH key). Raw outputs, including activations, fits, bundles and draws:
`~/SPAR/results/qwen8b_quick_20261125/`. Code: branch `eval-types-stage` at the commit that adds this file.
