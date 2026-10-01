# Temporal horizon robustness: Jared's working directory

Everything for the new SPAR experiments lives in `jared/`. Read in order.

| File | What it is |
|---|---|
| `00_README.md` | This index, plus lineage, environment, and model notes |
| `01_plan.md` | The research plan: question, R1-R5 robustness tests, decisions |
| `02_execution_plan.md` | The build plan: experiments E0-E5, tasks, status, how to ship |
| `03_baseline_qwen3.5-0.8b.md` | What the unmodified starter produced on the 0.8B model, and why its 100% reasoning score is an artifact |
| `spar_horizon/` | The package: prompts, model I/O, extraction, geometry, behavior, attribution, plots, run manifests |
| `experiments/` | One CLI per experiment (`exp0`-`exp5`) plus `run_all.sh` |
| `tests/` | 48 pytest tests on the pure-python parts and a tiny in-memory model |
| `horizon_geometry.py` / `.ipynb` | Notebook wrapper for E0; `build_notebook.py` regenerates the .ipynb |
| `requirements.txt` | What the vast.ai box needs on top of its torch image |
| `analysis/` | Post-analysis on synced results: cross-mode transfer, feature dimensionality, the 3D explorer (notebook + HTML), and `snapshot.py` |
| `snapshots/` | Frozen derived data per run (PCA coords, sweeps, manifests, figures; no activations) for comparing runs. One folder per snapshot with `snapshot.json` provenance |
| `starter/` | Untouched copies: the original starter script, its 116 prompts as markdown, and its figure |

## Lineage

`starter/spar_starter_geometry.py` is a 300-line distillation of the
intertemporal geometry pipeline in `../../temporal-awareness/` (Rios-Sialer
et al. 2026, temporal preference in Qwen3-4B-Instruct-2507):

| Starter piece | Prior-project source |
|---|---|
| Prompt constraint sentence | `src/intertemporal/formatting/configs/default_prompt_format.py:61` |
| PCA per (layer, token), Spearman vs log horizon | `src/intertemporal/geometry/`, `scripts/intertemporal/compute_geometry_analysis.py` |
| Turn-suffix and response positions | `src/intertemporal/common/semantic_positions.py` |
| Temporal reasoning / order / label stability | `scripts/intertemporal/coherent_behavior.py` |
| Attribution patching design (E5) | `src/attribution_patching/standard_attribution.py`, `attribution_metric.py` |

The paper localized the horizon signal to late-layer attention (L21, L24 in
Qwen3-4B) and steered it with CAA at L19-22.

## The prompt, in one line

Pick between a small near payout and a large far payout "for this time
horizon: X", with X log-swept from 30 seconds to 500 years. 96 horizon
prompts plus 20 no-horizon controls. Full list: `starter/starter_geometry_prompts.md`.

## Experiments

| | Name | Question | Output |
|---|---|---|---|
| E0 | baseline | Peak layer, probe layer, does the model read content | 2D + 3D PCA panels, activations.npz, behavior |
| E1 | nuisance | Is PC1 horizon, not reward or delay? Does the probe survive swap/relabel? | specificity heatmap, transfer numbers |
| E2 | paraphrase | Does the probe transfer across 6 constraint wordings? | 6x6 transfer matrix per layer |
| E3 | units | Does "1 year" land with "12 months" and "365 days"? | within/between spread ratio |
| E4 | perturb | Distractor, horizon-first, implicit horizon: probe transfer + behavior | table per variant |
| E5 | attribution | Which layer/position/component carries the horizon to the answer? | heatmaps, top-10 |

All take `--model`, `--thinking on|off`, `--limit N`, `--out DIR`, `--max-think-tokens`, `--batch-size`, `--prefill`. Each run
writes `manifest.json`. E1-E4 read E0's manifest for the layer choice, so E0
runs first (`run_all.sh` enforces this). E5 is thinking-off only.

## Environment

- **Local:** only `~/miniconda3/envs/mp` has torch, transformers (5.17), and
  pytest. Qwen3.5-0.8B runs on CPU there, slowly.
- **GPU:** one vast.ai RTX 4090 (24 GB) via `../../infra/vast.sh` and the
  `justfile` in the SPAR root. Defaults: `GPU_NAME=RTX_4090`,
  `MAX_PRICE=0.50`, `DISK_GB=40`.
- **HF token:** needed only for gated models. Qwen3-8B is ungated.

## Model notes

| Candidate | bf16 | Fits 4090 | Gated | Native think toggle | Verdict |
|---|---|---|---|---|---|
| **Qwen3-8B** | 16 GB | Yes | No | Yes | **Chosen** |
| Qwen3-14B | 28 GB | 8-bit only | No | Yes | Fallback if 8B is too weak |
| Llama-3.1-8B-Instruct | 16 GB | Yes | Yes | No | Ruled out by the thinking-on requirement |
| Gemma-2-9B-it | 18 GB | Tight | Yes | No | Same, plus needs eager attention |
| Qwen3.5-27B 4-bit | ~15 GB | Yes | No | Yes | Hybrid linear attention, hook tooling unsupported. Avoid |

## Running

```
# tests (fast, no model weights beyond the cached 0.8B tokenizer)
~/miniconda3/envs/mp/bin/python -m pytest -q tests

# local smoke: every experiment on the 0.8B, 6 prompts, thinking off (~8 min CPU)
MODEL=Qwen/Qwen3.5-0.8B LIMIT=6 MODES=off PY=~/miniconda3/envs/mp/bin/python bash experiments/run_all.sh

# local GPU: the intel_ai env has torch+xpu for the Intel Arc iGPU (device: xpu,
# bfloat16, ~7x faster than CPU on the 0.8B); load_model picks cuda > xpu > mps > cpu
PY=~/miniconda3/envs/intel_ai/bin/python bash experiments/run_all.sh   # same flags

# one experiment
python experiments/exp0_baseline.py --model Qwen/Qwen3-8B --thinking on --limit 12

# on the box: everything, both modes (E0 first); MAX_THINK and MODES are optional
MAX_THINK=3072 bash experiments/run_all.sh
```

Outputs go to `out/<exp>_<model>_think-<on|off>/` locally, or to
`$WORKSPACE/results/horizon/` on the box so `just sync` brings them home.

## Shipping (Phase C, not yet done)

```
just up
just push spar-planning-temporal-manifolds
just ssh "cd /workspace/SPAR/spar-planning-temporal-manifolds/jared && /venv/main/bin/python -m pip install -r requirements.txt"
just ssh "cd /workspace/SPAR/spar-planning-temporal-manifolds/jared && /venv/main/bin/python -m pytest -q tests"
just ssh "cd /workspace/SPAR/spar-planning-temporal-manifolds/jared && /venv/main/bin/python experiments/exp0_baseline.py --thinking off --limit 12"
# then the thinking-on smoke, then run_all.sh, then: just sync; just stop
```

`/venv/main/bin/python` is where the vast.ai image keeps torch; see
`infra/provision.sh` for the resolution order if that path differs.

## The answer position: prefill

Qwen3-8B with thinking off does not answer "a)"; 85% of the time it opens
with "To determine the best investment..." and names the label later. So the
assistant turn is prefilled with `I choose:` (setting `prefill`, `--prefill ""`
disables). With thinking off the prefill is part of the prompt, so the turn
suffix runs from `<|im_end|>` through the prefill. With thinking on the
prefill is appended after the (possibly force-closed) `</think>`. Any
formatting the model emits before the label, such as a markdown ` **`, is
skipped, so the "answer" position is the label itself. E5 teacher-forces the
clean run's lead tokens onto both sequences so its metric is read at the
position that predicts the label. Each extraction prints the token
distribution at the answer position; a WARNING appears if it is not a label.

## Kept positions

Every prompt keeps the same fixed-width layout, so no prompt is skipped for
its answer formatting or for a force-closed think block:

```
thinking on   (Qwen3-8B, prefill on; S = 5 suffix tokens on the 8B, 7 on the 0.8B)
[ suffix:    <|im_end|> \n <|im_start|> assistant \n (<think> \n) ]   region suffix   (0, S)
[ think:     think@0.25  think@0.5  think@0.75  think@1 ]              region think    (S, S+4)
[ pre:       </think>  \n\n  I  choose  : ]                             region pre      (S+4, S+9)
[ pre-label: the token right before the label (':' or a lead like ' **') ] region prelabel (S+9, S+10)
[ answer:    a  )  ... n_response tokens ]                               region answer   (S+10, S+16)

thinking off  (suffix runs through the prefill; think and pre are empty)
[ suffix ... I choose : ] [ pre-label ] [ answer ... ]
```

Think positions sit at `think_fractions` (default 0.25, 0.5, 0.75, 1.0) of
each prompt's own think span; f = 1 is the last token before `</think>`.
Their token differs per prompt, so they are labelled `think@0.25*`. The
regions are stored in `activations.npz` and the manifest; `n_suffix` is
kept as the index of the answer token for E1-E5. Thinking-on E0 also writes
`probe_track.png` (probe reading at every kept position), `dense_track.png`
(at every think token, from a second forward pass), and `label_probe.png`
(can the chosen label be read out before it is written?).

## Generation cache and progress

Every prompt's greedy generation is cached under `<out base>/gen_cache/`
keyed by the settings and input ids, so a rerun of E0 after an analysis
change skips the 20-minute thinking-on generation; `--no-cache` disables
it. Every loop shows a tqdm bar and each phase prints a timestamped line;
the manifest records `phases` (seconds per phase) and `code_hash`.

## Thinking mode on the 8B

Qwen3-8B reasons for 1.5k to 3k tokens on a normal prompt and loops without
end on absurd horizons (seconds, minutes), where it decides neither option
is feasible. The think block is therefore force-closed at
`max_think_tokens` (default 3072) and the run reports how many prompts were
forced, per prompt in `forced_mask`. Forced prompts are kept and flagged;
the manifest reports every think-position number for all prompts and for
the non-forced subset. Generation is batched; a thinking-on E0 takes about
35 minutes on the 4090 with `--skip-behavior` (which `run_all.sh` passes
for thinking on, since the behavior check regenerates the bank twice more
and its numbers are deterministic for a fixed bank and model).

## Snapshots: keeping runs comparable

After syncing a run home and computing its derived data:

```
python analysis/export_pca3.py results/horizon
python analysis/feature_dimensionality.py results/horizon
python analysis/cross_mode_transfer.py results/horizon
python analysis/snapshot.py results/horizon v2_prefill --note "prefill 'I choose:', 3072 think budget"
python analysis/snapshot.py --list
```

The explorer notebook reads a snapshot by name (`SNAPSHOT = ...`) and its last
section overlays two snapshots. Snapshots are small enough to commit (~16 MB);
the activation files are not copied and stay under `results/`.

## Smoke-test log

- 2026-09-30, mid-thinking positions: 63 tests green; 0.8B thinking-on E0 smoke (6 prompts, 32-token budget) writes 4 think columns, probe_track, dense_track, label_probe and a manifest with per-fraction numbers; thinking-off run_all smoke passes all six experiments; explorer export carries regions.
- 2026-09-21, 0.8B, thinking off, 6 prompts: suffix derived as 9 tokens
  (matches the starter), behavior reproduces the position bias.
- 2026-09-21, 0.8B, thinking on, 384 and 1024-token budgets: the think block
  never closes, but the reasoning text does read the horizon ("neither is
  feasible" at 300 seconds). Thinking-on results need the 8B.
- 2026-09-23, package refactor: E0 CLI reproduces the pre-refactor smoke
  exactly (suffix 9, peak 1.000 at layer 1, display layer 10, same behavior
  counts). 43 tests green.
- 2026-09-23, box, Qwen3-8B thinking on, 12 prompts, 1024 budget: all 12
  skipped, think block never closed. See "Thinking mode on the 8B".
- 2026-09-23, batched generation: token-identical to single-prompt on the
  0.8B (test_generation.py); force-close verified at a 24-token budget.
- 2026-09-23, box, Qwen3-8B thinking off, full run (archived locally as
  `results/horizon_v1_noprefill/`): probe R^2 0.9995 at L11, PC1 rho 0.95 at
  L16, E2 transfer 0.98-1.00, E3 ratio 0.13, E4 transfer 0.88-0.997, E5 33/48
  flips with attribution on the "months" token (L13-15) and the final
  position (L23-35). Behavior numbers invalid: the model wrote prose, not a
  label, in the 6 answer tokens. Fixed by the prefill; rerun pending.
- 2026-09-24, prefill + lead-token handling verified on the 0.8B in both
  modes; 48 tests green.
- 2026-09-23, all six experiments, 0.8B, 6 prompts, thinking off: every
  manifest written. E2 off-diagonal transfer 0.99 at layer 24 (6 prompts, so
  not meaningful yet). E5 uses 48 equal-length pairs on the full run; on the
  0.8B the a/b metric never flips, as expected from the position bias, but
  the top attribution sits on the turn token and the horizon unit token.
