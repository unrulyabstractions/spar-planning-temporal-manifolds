# Stage 3: analysis, split by label and by datatype

## Question

What happened, split by label (regime, unit, horizon, delay, truth, choice count, forced or
natural close) and by datatype (bank, number format, unit word)?

Everything here reads saved results. Nothing loads a model except the tokenizer in
`run_throughput.py`, `presentation_figures.py` and `causal_walkthrough.py`.

## Inputs

| Pattern | Produced by |
|---|---|
| `$RESULTS/exp0_<model>_think-<on\|off>[_<bank>]/{answers.json,activations.npz,manifest.json}` | stage 1 (`exp0_baseline`, `exp0_choice`) |
| `$RESULTS/exp9_*_units/`, `exp10_*`, `exp14_*_phrase*`, `exp15_*_scan`, `exp16_*_patch`, `exp17_*`, `exp18_*` | stage 1 (`exp14_phrase_states`) and stage 2 |
| `steering/inputs_*.{json,npz}` under `jared/` | stage 1 (`steering_prepare`, `patch_prepare`, `scan_prepare`) |

`$RESULTS` is defined in `../README.md` (What lives where).

## Outputs

| Pattern | Read by |
|---|---|
| `OUT_DIR/<script>.{json,txt}` (the `OUT_DIR` argument of the scripts that take one) | the results docs in `jared/` root (`24`, `13`) |
| `figures/*.png` and `<run>/figures/*.png` | `13_evidence.md`, the presentation |

Nothing downstream of this stage reads code outputs; the docs quote them.

## Contents

`split.py` is the shared loader. `load_runs(root, model, mode, banks)` reads the
`answers.json` of each `exp0_<model>_think-<mode>_<bank>` run into a list of dicts (a table)
with these columns, and `split_by(rows, column)` groups it:

| Column | Source in `answers.json` |
|---|---|
| `regime` | `regime` (none, near, both, null) |
| `unit` | `rendered_unit`, canonical (None on banks without a rendered unit) |
| `horizon` | `horizon` in years (None on null prompts) |
| `delay` | `delay_scale` (minutes-hours, days-weeks, months-years; only the crossed banks carry it) |
| `truth` | `truth` |
| `n_choices` | `n_options` (2 on the a/b banks, 4 or 8 on the choice banks) |
| `forced_close` | `forced` (the reasoning hit the token budget and was cut off) |
| `bank` | the run folder's name |
| `number_format` | "digits" or "written", from the horizon phrase (`duration_id`, else its natural rendering) |
| `chose_near` | `label` against `near_label` (a/b) or `near_labels` (choice banks) |

Scripts that predate `split.py` keep their own loaders and logic. The "splits on" columns
below say which labels or datatypes each script groups by.

### by_label/

| Script | What it does | Splits on | Example |
|---|---|---|---|
| `choice_by_label.py` | near share by any label, natural against forced closes (uses `split.py`) | regime, unit, delay, truth, choice count; forced close | `python -m pipeline.s3_analysis.by_label.choice_by_label $RESULTS --label regime` |
| `regime_comparison.py` | is there an explicit horizon-vs-delay comparison before the hand-off | regime; delay | `python -m pipeline.s3_analysis.by_label.regime_comparison $RESULTS` |
| `confound_sweep.py` | what else moves the horizon code within a duration | regime, option order, delay | `python -m pipeline.s3_analysis.by_label.confound_sweep $RESULTS` |
| `confound_robustness.py` | how much each confound depends on the readout | regime, option order | `python -m pipeline.s3_analysis.by_label.confound_robustness $RESULTS` |
| `delay_margin.py` | does the code read the horizon relative to the delays | delay, regime | `python -m pipeline.s3_analysis.by_label.delay_margin $RESULTS` |
| `far_margin.py` | the far-delay version of `delay_margin` | delay (far) | `python -m pipeline.s3_analysis.by_label.far_margin $RESULTS` |
| `delay_position.py` | which delay pulls the code, the near one or the one listed nearer | delay, regime none | `python -m pipeline.s3_analysis.by_label.delay_position $RESULTS` |
| `delay_step.py` | step or slope where a delay crosses the horizon | delay, regime | `python -m pipeline.s3_analysis.by_label.delay_step $RESULTS` |
| `axis_delay_purity.py` | which steering axis moves the horizon without moving the delays | delay, horizon | `python -m pipeline.s3_analysis.by_label.axis_delay_purity $RESULTS` |
| `behaviour_step.py` | is the choice a threshold on the delays or graded | delay, choice | `python -m pipeline.s3_analysis.by_label.behaviour_step $RESULTS` |
| `choice_from_code.py` | does the encoded horizon predict the choice beyond the true horizon | choice, horizon, regime | `python -m pipeline.s3_analysis.by_label.choice_from_code $RESULTS` |
| `decision_emergence.py` | when is the choice readable beyond the prompt surface | regime, choice | `python -m pipeline.s3_analysis.by_label.decision_emergence $RESULTS OUT_DIR` |
| `breakeven_rate.py` | does the choice follow the break-even reinvestment rate | regime both, choice | `python -m pipeline.s3_analysis.by_label.breakeven_rate $RESULTS OUT_DIR` |
| `breakeven_decode.py` | is the break-even rate computed before reasoning | regime both, choice | `python -m pipeline.s3_analysis.by_label.breakeven_decode $RESULTS OUT_DIR` |
| `heuristic_disposition.py` | is the arithmetic shortcut set before reasoning | regime both, choice | `python -m pipeline.s3_analysis.by_label.heuristic_disposition $RESULTS OUT_DIR` |
| `trace_audit.py` | what the reasoning says and how it lines up with the choice | regime, choice, forced close, bank | `python -m pipeline.s3_analysis.by_label.trace_audit $RESULTS OUT_DIR` |
| `trace_coding_summary.py` | arithmetic in the both-regime reasoning and the choice | regime both, choice, forced close | `python -m pipeline.s3_analysis.by_label.trace_coding_summary $RESULTS OUT_DIR` |
| `rerun_stability.py` | how often a rerun of the same prompt flips the choice | regime, forced close, choice | `python -m pipeline.s3_analysis.by_label.rerun_stability $RESULTS OUT_DIR` |
| `exp14_analysis.py` | did steering the horizon phrase move the reasoning and the choice (spec 21) | horizon shift, choice | `python -m pipeline.s3_analysis.by_label.exp14_analysis <dir>` |
| `choice_structure.py` | probes and geometry on choice-bank chunks | choice count (4, 8), truth | `python -m pipeline.s3_analysis.by_label.choice_structure $RESULTS/exp0_Qwen3-8B_think-on_choice4` |

### by_datatype/

| Script | What it does | Splits on | Example |
|---|---|---|---|
| `choice_by_datatype.py` | near share by bank, number format or unit word (uses `split.py`) | bank, number format, unit word | `python -m pipeline.s3_analysis.by_datatype.choice_by_datatype $RESULTS --datatype bank` |
| `unit_overlap.py` | horizon or unit word, E0 on the units bank (spec 14) | unit word, duration; units bank | `python -m pipeline.s3_analysis.by_datatype.unit_overlap $RESULTS` |
| `unit_readout.py` | does the probe read the duration, the number or the unit word | unit word, number; units bank | `python -m pipeline.s3_analysis.by_datatype.unit_readout $RESULTS` |
| `unit_staircase.py` | does the code step at unit boundaries | natural unit; full bank | `python -m pipeline.s3_analysis.by_datatype.unit_staircase $RESULTS` |
| `unit_number_rsa.py` | duration, unit word or written number in the prompt-end geometry (E9) | unit word, number | `python -m pipeline.s3_analysis.by_datatype.unit_number_rsa $RESULTS OUT_DIR` |
| `numeral_digits.py` | is the shared numeral code digit-structured | number format (digits) | `python -m pipeline.s3_analysis.by_datatype.numeral_digits $RESULTS OUT_DIR` |
| `units_analysis.py` | E9: does the readout track the duration or the unit word | unit word | `python -m pipeline.s3_analysis.by_datatype.units_analysis $RESULTS` |
| `written_unit.py` | which unit the reasoning writes the horizon in (number parsing also used by other scripts) | unit word, number format in the trace; units bank | `python -m pipeline.s3_analysis.by_datatype.written_unit $RESULTS` |
| `run_throughput.py` | measured throughput of earlier box runs and prompt token counts | bank | `python -m pipeline.s3_analysis.by_datatype.run_throughput $RESULTS` |
| `big_experiment_costs.py` | cost model for the first big experiment (`big_experiment_plan.md`) | bank | `python -m pipeline.s3_analysis.by_datatype.big_experiment_costs` |

### figures/

Scripts that only draw, from saved results.

| Script | What it does | Example |
|---|---|---|
| `evidence_figures.py` | figures of `13_evidence.md` that no other script draws | `python -m pipeline.s3_analysis.figures.evidence_figures $RESULTS` |
| `presentation_figures.py` | presentation figures (decodable is not causal, and others) | `python -m pipeline.s3_analysis.figures.presentation_figures $RESULTS` |
| `choice4_implied_figures.py` | figures of the choice4 think-budget and implied-horizon experiments (specs 28, 29) | `python -m pipeline.s3_analysis.figures.choice4_implied_figures [results dir] [output dir]` |
| `patch_figures.py` | figures of the phrase number patch (spec 27) | `python -m pipeline.s3_analysis.figures.patch_figures --patch-dir <dir> --phrase-dir <dir>` |
| `causal_walkthrough.py` | figures and numbers for `09_causal_tests_explained.md` | `python -m pipeline.s3_analysis.figures.causal_walkthrough $RESULTS` |

## How to run

Locally, from `jared/`, with the `mp` env (`~/miniconda3/envs/mp/bin/python`), on results synced
from the box (`just sync`). Most scripts need only `answers.json` and `activations.npz`; a few
load activations of several runs and take minutes. No script here needs the GPU, and there are
no queue files for this stage. The tests are in `tests/` and run with the fast suite:
`~/miniconda3/envs/mp/bin/python -m pytest -q tests --ignore=tests/test_generation.py`.

## Moved from

Everything here came from `jared/analysis/<name>.py`, unchanged except `sys.path.insert` removed,
imports rewritten to `pipeline.<stage>...`, usage lines changed to `python -m ...`, and the four
`Path(__file__)` anchors (`run_throughput`, `choice4_implied_figures`, `presentation_figures`,
`causal_walkthrough`) re-pointed at `jared/`. New: `split.py`, `by_label/choice_by_label.py`,
`by_datatype/choice_by_datatype.py`. Old paths: `git log --follow <new path>`.

Sorting calls: `run_throughput` and `big_experiment_costs` are in `by_datatype` (per bank),
`exp14_analysis` and `choice_structure` in `by_label`, `causal_walkthrough` in `figures`.
`run_records` lives in `spar_horizon/e0_runs.py`; `written_unit` re-exports it.

## Docs

The project-wide docs (`12`, `13`, `23`, `24`, `31`, `big_experiment_plan.md`) stay in `jared/` root
and are listed in `00_README.md`. Stage 1 and 2 specs and results: `docs/INDEX.md` in those stages.

## Status and known gaps

- `split.py` reads only `answers.json`. It has no delays in years or the near and far reward
  (those are in the records `written_unit.run_records` rebuilds from the manifest), so `delay`
  is the coarse `delay_scale` and exists only on the crossed banks (units, fill).
- Every current prompt bank writes the horizon with digits ("36 months"); "written" in
  `number_format` is supported but empty on today's data. Number words appear only in the
  reasoning, which `written_unit.py` parses.
- The older scripts keep their own loaders (`unit_overlap.load_run`, `confound_sweep.dataset`,
  `written_unit.run_records`); moving them onto `split.py` is not done.
- `unit_number_rsa.py` and `confound_sweep.py` keep an unused import (`UNIT_YEARS`, `Horizon`)
  from before the move.
