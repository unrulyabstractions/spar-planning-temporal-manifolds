# Gemma 4 31B, first evaluation: quick test (2026-11-24)

**Question.** (A) Do the extraction side's steering methods move Gemma's horizon choice on the T0 bank more than random
edits of the same size do? (B) Does the effect carry over to the eval type most expected to transfer (`system_prompt`)?

**Short answer.** No, not yet. At the calibrated strength CAA moves the answer in the right direction (+1.5 logits
toward the far option), but random directions with the activations' covariance move it as much (p = 0.30). On
`system_prompt` the effect is half as large and again not distinguishable from random (p = 0.35).

## Setup

- Alan's hand-over (`CodeReclaimers/spar-steering-gemma-4-31b`): baseline run on all 700 T0 prompts (thinking on,
  sampled), layer 38, label site, bundles as fitted; every token edited (the bundles' default).
- Readout: replay, i.e. the model's own reasoning held fixed and the answer logits read under the edit, in bf16.
- Strength frozen on held-out extraction prompts (fold 0, 38 prompts): x = 0.1 × typical norm, the smallest strength
  with a 1-logit effect and no loss of label mass. From x = 0.4 the label mass drops (the model breaks).
- Random baseline: 19 directions matched to the activations' covariance (`covariance`), the hardest of the cheap
  families. On Gemma these reach cosine 0.76 with CAA (`extraction_design/README.md` §10.1).
- Files: `gemma-4-31B-it__t0_s0__quick_report.md` (T0 report card) and `…__quick_transfer.md` (transfer). The
  report card's last line ("Plumbing run: the model is small…") is boilerplate from `analyze.py` and does not apply.

## Results

| | prompts | meaning (logits) | p vs covariance | switch-point shift (decades) |
|---|---|---|---|---|
| CAA, T0 verification grid | 200 | +1.54 | 0.30 | +0.01 |
| CAA, `system_prompt` | 40 | +0.74 | 0.35 | +0.00 |
| "be long-term-minded" appended, `system_prompt` | 40 | +0.31 | – | – |
| "as if the horizon were 100× longer", `system_prompt` | 40 | +0.40 | – | – |
| neutral sentence, `system_prompt` | 40 | −0.04 | – | – |

- **CAA moves the answer but not the horizon.** The switch point (the horizon at which the model turns from near to
  far) does not move, and the part of the effect that is the same at +x and −x (disruption, +1.20) is nearly as large
  as the steering part.
- **curve-snap mostly disrupts.** It pushes the wrong way (−1.05), favours one answer letter (+1.80), and flattens the
  choice curve (slope 5.1 → 0.8 / 0.07): the model stops responding to the horizon. No random baseline was run for it.
- **Steered generation (16 prompts):** P(far) 0.25 unsteered, 0.33 with CAA +x, 0.25 with CAA −x. Too few prompts
  to read.

## Caveats

- One layer, one site, every token edited, reasoning held fixed. Alan found that by the prelabel token Gemma's
  choice is already settled (≥ 13 logits), which fits an edit at the answer having little room.
- The covariance baseline is demanding: the result says CAA is no better than a random high-variance direction of
  the same size, not that it does nothing. The isotropic and shuffled-label baselines were not run.
- K = 19 draws (smallest possible p = 0.05); bf16 replay (about 0.1–0.3 logit of rounding per prompt, small next
  to the baseline's spread).

## Next

1. Edit earlier (`think_end`, `prompt_end`), where the edit can change the reasoning.
2. Add the isotropic and shuffled-refit baselines, to show how much the choice of baseline decides the verdict.
3. The within-regime CAA bundles, to separate "horizon" from "chose far".

Run: one A100 80 GB, about 2 GPU-hours, about $2.30. Code: branch `eval-types-stage` at the commit that adds this file.
