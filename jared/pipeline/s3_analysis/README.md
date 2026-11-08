# Stage 3: analysis, split by label and by datatype

## Question

What happened, split by label (regime, unit, horizon, delay, truth, choice count, forced or
natural close) and by datatype (bank, number format, unit word)?

Everything here reads saved results. Nothing loads a model except the tokenizer in
`run_throughput.py`, `presentation_figures.py` and `causal_walkthrough.py`.

## Inputs

| Pattern | Produced by |
|---|---|
| `out/exp0_<model>_think-<on\|off>[_<bank>]/{answers.json,activations.npz,manifest.json}` (on the box: `$WORKSPACE/results/horizon/`) | stage 1 (`exp0_baseline`, `exp0_choice`) |
| `out/exp9_*_units/`, `out/exp10_*`, `out/exp14_*_phrase*`, `out/exp16_*_patch`, `out/exp17_*`, `out/exp18_*` | stage 1 (`exp14_phrase_states`) and stage 2 |
| `steering/exp*_*/` (steering, patching and scan results) | stage 2 |
| `steering/inputs_*.{json,npz}` (steering inputs) | stage 1 (`steering_prepare`, `patch_prepare`, `scan_prepare`) |

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
| `choice_by_label.py` | near share by any label, natural against forced closes (uses `split.py`) | regime, unit, delay, truth, choice count; forced close | `python -m pipeline.s3_analysis.by_label.choice_by_label results/horizon --label regime` |
| `regime_comparison.py` | is there an explicit horizon-vs-delay comparison before the hand-off | regime; delay | `python -m pipeline.s3_analysis.by_label.regime_comparison results/horizon` |
| `confound_sweep.py` | what else moves the horizon code within a duration | regime, option order, delay | `python -m pipeline.s3_analysis.by_label.confound_sweep results/horizon` |
| `confound_robustness.py` | how much each confound depends on the readout | regime, option order | `python -m pipeline.s3_analysis.by_label.confound_robustness results/horizon` |
| `delay_margin.py` | does the code read the horizon relative to the delays | delay, regime | `python -m pipeline.s3_analysis.by_label.delay_margin results/horizon` |
| `far_margin.py` | the far-delay version of `delay_margin` | delay (far) | `python -m pipeline.s3_analysis.by_label.far_margin results/horizon` |
| `delay_position.py` | which delay pulls the code, the near one or the one listed nearer | delay, regime none | `python -m pipeline.s3_analysis.by_label.delay_position results/horizon` |
| `delay_step.py` | step or slope where a delay crosses the horizon | delay, regime | `python -m pipeline.s3_analysis.by_label.delay_step results/horizon` |
| `axis_delay_purity.py` | which steering axis moves the horizon without moving the delays | delay, horizon | `python -m pipeline.s3_analysis.by_label.axis_delay_purity results/horizon` |
| `behaviour_step.py` | is the choice a threshold on the delays or graded | delay, choice | `python -m pipeline.s3_analysis.by_label.behaviour_step results/horizon` |
| `choice_from_code.py` | does the encoded horizon predict the choice beyond the true horizon | choice, horizon, regime | `python -m pipeline.s3_analysis.by_label.choice_from_code results/horizon` |
| `decision_emergence.py` | when is the choice readable beyond the prompt surface | regime, choice | `python -m pipeline.s3_analysis.by_label.decision_emergence results/horizon OUT_DIR` |
| `breakeven_rate.py` | does the choice follow the break-even reinvestment rate | regime both, choice | `python -m pipeline.s3_analysis.by_label.breakeven_rate results/horizon OUT_DIR` |
| `breakeven_decode.py` | is the break-even rate computed before reasoning | regime both, choice | `python -m pipeline.s3_analysis.by_label.breakeven_decode results/horizon OUT_DIR` |
| `heuristic_disposition.py` | is the arithmetic shortcut set before reasoning | regime both, choice | `python -m pipeline.s3_analysis.by_label.heuristic_disposition results/horizon OUT_DIR` |
| `trace_audit.py` | what the reasoning says and how it lines up with the choice | regime, choice, forced close, bank | `python -m pipeline.s3_analysis.by_label.trace_audit results/horizon OUT_DIR` |
| `trace_coding_summary.py` | arithmetic in the both-regime reasoning and the choice | regime both, choice, forced close | `python -m pipeline.s3_analysis.by_label.trace_coding_summary results/horizon OUT_DIR` |
| `rerun_stability.py` | how often a rerun of the same prompt flips the choice | regime, forced close, choice | `python -m pipeline.s3_analysis.by_label.rerun_stability results/horizon OUT_DIR` |
| `exp14_analysis.py` | did steering the horizon phrase move the reasoning and the choice (spec 21) | horizon shift, choice | `python -m pipeline.s3_analysis.by_label.exp14_analysis <dir>` |
| `choice_structure.py` | probes and geometry on choice-bank chunks | choice count (4, 8), truth | `python -m pipeline.s3_analysis.by_label.choice_structure results/horizon/exp0_Qwen3-8B_think-on_choice4` |

### by_datatype/

| Script | What it does | Splits on | Example |
|---|---|---|---|
| `choice_by_datatype.py` | near share by bank, number format or unit word (uses `split.py`) | bank, number format, unit word | `python -m pipeline.s3_analysis.by_datatype.choice_by_datatype results/horizon --datatype bank` |
| `unit_overlap.py` | horizon or unit word, E0 on the units bank (spec 14) | unit word, duration; units bank | `python -m pipeline.s3_analysis.by_datatype.unit_overlap results/horizon` |
| `unit_readout.py` | does the probe read the duration, the number or the unit word | unit word, number; units bank | `python -m pipeline.s3_analysis.by_datatype.unit_readout results/horizon` |
| `unit_staircase.py` | does the code step at unit boundaries | natural unit; full bank | `python -m pipeline.s3_analysis.by_datatype.unit_staircase results/horizon` |
| `unit_number_rsa.py` | duration, unit word or written number in the prompt-end geometry (E9) | unit word, number | `python -m pipeline.s3_analysis.by_datatype.unit_number_rsa results/horizon OUT_DIR` |
| `numeral_digits.py` | is the shared numeral code digit-structured | number format (digits) | `python -m pipeline.s3_analysis.by_datatype.numeral_digits results/horizon OUT_DIR` |
| `units_analysis.py` | E9: does the readout track the duration or the unit word | unit word | `python -m pipeline.s3_analysis.by_datatype.units_analysis results/horizon` |
| `written_unit.py` | which unit the reasoning writes the horizon in (number parsing also used by other scripts) | unit word, number format in the trace; units bank | `python -m pipeline.s3_analysis.by_datatype.written_unit results/horizon` |
| `run_throughput.py` | measured throughput of earlier box runs and prompt token counts | bank | `python -m pipeline.s3_analysis.by_datatype.run_throughput ../../results/horizon` |
| `big_experiment_costs.py` | cost model for the first big experiment (`big_experiment_plan.md`) | bank | `python -m pipeline.s3_analysis.by_datatype.big_experiment_costs` |

### figures/

Scripts that only draw, from saved results.

| Script | What it does | Example |
|---|---|---|
| `evidence_figures.py` | figures of `13_evidence.md` that no other script draws | `python -m pipeline.s3_analysis.figures.evidence_figures results/horizon` |
| `presentation_figures.py` | presentation figures (decodable is not causal, and others) | `python -m pipeline.s3_analysis.figures.presentation_figures results/horizon` |
| `choice4_implied_figures.py` | figures of the choice4 think-budget and implied-horizon experiments (specs 28, 29) | `python -m pipeline.s3_analysis.figures.choice4_implied_figures [results dir] [output dir]` |
| `patch_figures.py` | figures of the phrase number patch (spec 27) | `python -m pipeline.s3_analysis.figures.patch_figures --patch-dir <dir> --phrase-dir <dir>` |
| `causal_walkthrough.py` | figures and numbers for `09_causal_tests_explained.md` | `python -m pipeline.s3_analysis.figures.causal_walkthrough results/horizon` |

## How to run

Locally, from `jared/`, with the `mp` env (`~/miniconda3/envs/mp/bin/python`), on results synced
from the box (`just sync`). Most scripts need only `answers.json` and `activations.npz`; a few
load activations of several runs and take minutes. No script here needs the GPU, and there are
no queue files for this stage. The tests are in `tests/` and run with the fast suite:
`~/miniconda3/envs/mp/bin/python -m pytest -q tests --ignore=tests/test_generation.py`.

## Moved from

All from `jared/analysis/<name>.py`; `<name>.py` is unchanged unless noted.

| Old | New |
|---|---|
| `analysis/{regime_comparison,confound_sweep,confound_robustness,delay_margin,far_margin,delay_position,delay_step,axis_delay_purity,behaviour_step,choice_from_code,decision_emergence,breakeven_rate,breakeven_decode,heuristic_disposition,trace_audit,trace_coding_summary,rerun_stability,exp14_analysis,choice_structure}.py` | `pipeline/s3_analysis/by_label/` |
| `analysis/{unit_overlap,unit_readout,unit_staircase,unit_number_rsa,numeral_digits,units_analysis,written_unit,run_throughput,big_experiment_costs}.py` | `pipeline/s3_analysis/by_datatype/` |
| `analysis/{evidence_figures,presentation_figures,choice4_implied_figures,patch_figures,causal_walkthrough}.py` | `pipeline/s3_analysis/figures/` |
| (new) | `pipeline/s3_analysis/split.py`, `by_label/choice_by_label.py`, `by_datatype/choice_by_datatype.py` |

Sorting calls that the spec left open: `run_throughput` and `big_experiment_costs` go to
`by_datatype` (they are per bank); `exp14_analysis` to `by_label` (horizon shift against
choice); `choice_structure` to `by_label` (it splits the choice banks by choice count);
`causal_walkthrough` to `figures`. `written_unit` is imported by most of the by_label scripts
for `run_records`.

Edits in the moved files: `sys.path.insert` removed, imports rewritten to
`pipeline.<stage>...`, `python analysis/x.py` usage lines changed to `python -m ...`, and the
four `Path(__file__)` anchors (`run_throughput`, `choice4_implied_figures`,
`presentation_figures`, `causal_walkthrough`) re-pointed at `jared/` from the new depth.

## Docs

The numbered docs that index and summarise the whole project stay in `jared/` root; this stage
owns only the index (the root `00_README.md` and `01_plan.md` paths are updated by the lead).

| Doc | What it holds |
|---|---|
| `00_README.md` | entry point: what is in the working directory and how to read it |
| `01_plan.md` | status, decisions and the audit trail of claims |
| `12_results_local_checks.md` | two local checks on saved activations, 8B and 14B |
| `13_evidence.md` | each claim, its strength, and the plot behind it |
| `23_literature.md` | where the project sits against the literature |
| `24_results_overnight_autoresearch.md` | results of the 2026-11-03 overnight analyses (`by_label/` trace, break-even and decision scripts, `by_datatype/` unit-number scripts) |
| `31_goals_next_specs.md` | goals to draft as the next specs |
| `big_experiment_plan.md` | costs from measured runs and a recommendation (`big_experiment_costs.py`); not numbered |
| `README.md` | short repo-level readme |

The specs and results of stages 1 and 2 are in those stages' `docs/`.

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
