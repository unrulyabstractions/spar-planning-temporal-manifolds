---
# License of this dataset card and the derived data (same as the project's other datasets). The activations are
# derived from Qwen/Qwen3-14B (Apache-2.0).
license: mit
pretty_name: "PTM multi-turn planning: Qwen3-14B residual-stream activations at turn boundaries of 5-step plans"
language:
- en
tags:
- interpretability
- mechanistic-interpretability
- activations
- residual-stream
- probing
- time-horizon
- planning
- multi-turn
- qwen3
size_categories:
- 1K<n<10K
configs:
- config_name: turns
  default: true
  data_files:
  - split: turns
    path: runs/qwen3-14b_mtp_s0/index.parquet
- config_name: conversations
  data_files:
  - split: conversations
    path: runs/qwen3-14b_mtp_s0/conversations.parquet
---

# Multi-turn planning: Qwen3-14B activations at the turn boundaries of five-step plans

Residual-stream activations of **Qwen3-14B** while it writes a **five-step plan over seven chat turns**, for 480
conversations (3,360 assistant turns). It comes from the *multi-turn planning* experiment of the SPAR (fall 2026)
project **Planning Temporal Manifolds** (PTM). Most conversations state a target time horizon for the plan
(1 week to 50 years), and the model writes a horizon for every step. The data lets you ask whether the horizon the
model is planning over, and the horizon of the step it is about to write, can be read from its activations at
the **turn-boundary tokens** before each reply.

This card documents the data only. Design, code and analysis live in the experiment folder
[`multiturn_planning/`](https://github.com/unrulyabstractions/spar-planning-temporal-manifolds/tree/multiturn-planning/multiturn_planning)
of the project repository (branch `multiturn-planning`). A results explainer with figures and a methods primer is
`multiturn_planning/explainer/index.html` there.

## Background

PTM asks whether the time horizon an LLM is planning over can be read from its residual stream at turn-boundary
tokens. It extends *Temporal Concepts and their Shape in Large Language Models* (arXiv:2606.05194), which found that
for single decisions with a stated horizon, the horizon lies along an ordered one-dimensional manifold (first
principal component vs. log horizon, Spearman |ρ| ≈ 0.95). Teammates replicated this on Qwen3-14B (Alan) and
Qwen3-8B (Jared). This dataset extends the setting from one decision to a plan written over several turns.

## Protocol

- A fixed **system prompt** (the same in every conversation; text in `capture_meta.json`) sets the format: the first
  reply is an outline of exactly 5 step titles, with no timings; each user `Continue` gets one step, written as
  `Step <n>: <title>` / `Time horizon: <how far from today this step will be complete, one duration>` /
  `Details: …`; after step 5, `Continue` gets `Plan Completed`. That makes **7 assistant turns** per conversation:
  turn 1 = outline, turns 2–6 = steps 1–5, turn 7 = `Plan Completed`.
- The **first user message** = one of 12 goals + `Help me make a plan.` + (optionally) a horizon sentence, e.g.
  *"Our company wants to cut its carbon emissions. Help me make a plan. The time horizon for this plan is 25 years."*
- **With a target** (`condition == "horizon"`, 288 conversations): 12 goals × 8 targets (1 week, 1 month, 3 months,
  1 year, 3 years, 10 years, 25 years, 50 years) × 3 wordings of the horizon sentence (`within`: "The plan should
  achieve this within {h}.", `horizon`: "The time horizon for this plan is {h}.", `from_today`: "All of it needs to
  happen within {h} from today."). Greedy decoding.
- **Without a target** (`condition == "none"`, 192 conversations): the same 12 goals with no horizon sentence, so the
  model chooses every horizon itself. There is one prompt per goal, so plans are **sampled** (Qwen3's non-thinking
  settings: temperature 0.7, top-p 0.8, top-k 20), 16 per goal (`sample` 0–15).
- Thinking off (Qwen3's template adds an empty think block to the current turn).
- **Step replies are prefilled** with `Step <n>:`: the model continues from there, so it cannot skip or renumber a
  step. The prefill comes after the boundary window P0–P8 below, so the activations there are unaffected.
- **Guard**: a step reply that starts a second step or appends "Plan Completed" is cut there (`cut`; 6 replies).
- Every turn is captured by re-running exactly the token sequence it was generated from (teacher-forced, unbatched,
  no padding). Qwen3's template drops the empty think block from *earlier* turns in the history, so a re-tokenised
  transcript would not match the sequence the model actually saw.

## Contents

| Path | What |
|---|---|
| `runs/qwen3-14b_mtp_s0/acts_NNNN.safetensors` | activations: 14 shards, tensors `acts` (float16, `[rows, 11, 13, 5120]`) and `valid` (bool, `[rows, 13]`) |
| `runs/qwen3-14b_mtp_s0/index.parquet` | one row per (conversation, turn): 3,360 rows with the reply text, parsed step horizon and token positions |
| `runs/qwen3-14b_mtp_s0/conversations.parquet` | one row per conversation: 480 rows with the first user message |
| `runs/qwen3-14b_mtp_s0/capture_meta.json` | model + revision, git commit, torch/transformers, GPU, layers, positions, system prompt, prefill, max new tokens per turn, sampling settings, wall time |
| `runs/qwen3-14b_mtp_s0/adherence.txt` | format-adherence report written on the capture machine |
| `runs/qwen3-14b_mtp_s0/pipeline.log`, `runs/qwen3-14b_mtp_s0.{capture,evaluate}.log` | logs from the capture machine (its evaluate step was a quick sanity pass, stopped early; the log is empty) |
| `results/qwen3-14b_mtp_s0/` | the full analysis, run locally afterwards: `A_persistence.csv`, `B_next_step.csv`, `C_first_turn.csv`, `D_transfer.csv`, `behavior.json`, `summary.md` |
| `runs/qwen3-14b_mtp_s0.evaluate_local.log` | log of that analysis |
| `runs/qwen3-14b_mtp_s0/SHA256SUMS.release` | **checksums of every file in this dataset** except itself, `SHA256SUMS` and this card |
| `runs/qwen3-14b_mtp_s0/SHA256SUMS` | the capture machine's record, used to verify the copy back. Its `results/…/behavior.json` line is the quick sanity version, later replaced by the full analysis, so check a download with `SHA256SUMS.release` |

About 4.9 GB in total, almost all of it activations. This is the **whole capture**, not a subset. The layout mirrors
the experiment folder, so from the root of a download `sha256sum -c runs/qwen3-14b_mtp_s0/SHA256SUMS.release`
checks it, and `python scripts/evaluate.py runs/qwen3-14b_mtp_s0` (with the repository's `multiturn_planning/`
code) reruns the analysis (about 13 min on 12 CPU cores).

## Provenance

| | |
|---|---|
| Code | capture at commit `a13c941` (branch `multiturn-planning`); analysis at commit `54d15a4` |
| Model | `Qwen/Qwen3-14B`, revision `40c069824f4251a91eefaf281ebe4c544efd3e18`, bf16, thinking disabled |
| Decoding | batch 16, left-padded for generation; greedy (with target) or sampled (without); max new tokens 200 (outline), 220 (steps), 16 (last turn) |
| Activations | from a separate unpadded teacher-forced pass per turn, stored as float16 |
| Software | Python 3.12, torch 2.11.0+cu128, transformers 5.17.0 |
| Hardware | 1× NVIDIA RTX PRO 5000 Blackwell (48 GB), rented on vast.ai; capture took 19 min (1,143 s) |
| Date | 2026-09-29 |

## Activations

- `acts`: shape `[rows, 11, 13, 5120]` = (turn row, layer, position, hidden dimension), **float16**. Shards hold 256
  rows each (the last one 32), 3,360 rows in total.
- **Row mapping**: the `index.parquet` columns `shard` and `row_in_shard` locate each turn; the global row is
  `256 * shard + row_in_shard`. Rows are in **capture order** (turn 1 of all conversations, then turn 2, …), not
  grouped by conversation.
- `valid[row, position]` is False where a position does not exist in that turn (H and V outside step turns; E when a
  reply was truncated). Those entries hold the activation at token 0 and must be ignored.
- **Layers** (second axis): 0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40 of 40. Layer 0 = the embedding output; layer
  *l* = the residual stream after the *l*-th decoder block; no final norm. Layers 16, 24 and 32 are the proposal's
  0.4 / 0.6 / 0.8 × depth.
- **Positions** (third axis, in this order):

  | name | token(s) | what is in context there |
  |---|---|---|
  | `P0` … `P8` | the boundary window before each reply: `<\|im_end\|>` `\n` `<\|im_start\|>` `assistant` `\n` `<think>` `\n\n` `</think>` `\n\n` | everything up to the end of the user's turn, **not** the reply being written |
  | `U` | last token of the user's turn (`Continue`, or the last token of the first message) | the same |
  | `H` | the token holding the colon of `Time horizon:` inside a step reply | the step's title |
  | `V` | the token holding the last character of the horizon value (e.g. `months` in `6 months`) | the value itself (a positive control) |
  | `E` | the `<\|im_end\|>` closing the reply | the whole reply |

  At layer 0 the P0–P8 tokens are the same in every conversation, so their activations are identical across
  conversations: a built-in sanity check (any probe there must fail).
- Exact token indices per turn are in `index.parquet` → `positions` (JSON; `null` where absent).

### Columns of `index.parquet`

| Column(s) | Meaning |
|---|---|
| `conv_id` | conversation key, e.g. `bakery__within__1week` (with target) or `bakery__none__s0` (without) |
| `scenario`, `condition`, `wording` | goal (12); `horizon` / `none`; wording of the horizon sentence (`within`, `horizon`, `from_today`; empty without a target) |
| `h_target_text`, `h_target_years` | the stated target (empty without a target); 1 week = 1/52 year, 1 month = 1/12 |
| `sample`, `greedy` | sample number of a no-target plan; greedy or sampled |
| `turn`, `kind` | 1–7; `outline` (turn 1), `step` (turns 2–6) or `done` (turn 7) |
| `reply` | the assistant's reply in that turn (for steps, including the prefilled `Step <n>:`) |
| `step_no`, `h_step_text`, `h_step_years` | parsed step number and step horizon (the first duration after `Time horizon:`; a range gives its upper end). Empty when unparseable, e.g. "Ongoing" (3.2% of steps, mostly step 5) |
| `truncated`, `cut` | the reply hit the token limit; the guard cut the reply |
| `outline_mentions_time`, `outline_full` | turn 1 only: the outline mentions a duration; it wrote full steps (both 0% in this run) |
| `prompt_len`, `n_reply` | tokens before the reply; tokens in the reply |
| `shard`, `row_in_shard`, `positions` | where the activations are; token index of every named position |

`conversations.parquet` has the per-conversation fields plus `first_user`, the full first user message.

### Loading

```python
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from safetensors.numpy import load_file

root = Path(snapshot_download("anicola/ptm-multiturn-planning-qwen3-14b", repo_type="dataset"))  # ~4.9 GB
run = root / "runs/qwen3-14b_mtp_s0"
shards = [load_file(str(p)) for p in sorted(run.glob("acts_*.safetensors"))]
acts = np.concatenate([s["acts"] for s in shards])            # (3360, 11, 13, 5120) float16, ~4.6 GB in memory
valid = np.concatenate([s["valid"] for s in shards])          # (3360, 13) bool
df = pd.read_parquet(run / "index.parquet")
df["row"] = 256 * df["shard"] + df["row_in_shard"]

LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40]
POS = ["P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "U", "H", "V", "E"]
# Example: the target horizon at the boundary before the outline (turn 1), layer 32, token P4
t1 = df[(df.turn == 1) & (df.condition == "horizon")]
X = acts[t1.row.values, LAYERS.index(32), POS.index("P4")].astype(np.float32)   # (288, 5120)
y = np.log10(t1.h_target_years.values)
```

With the repository's code, `mtp.analysis.load_run(run)` returns the same table (plus `row`, `log_h_step` and
`log_h_target`), `acts`, `valid` and the metadata.

## Results in brief

Probes are ridge regressions to log10 years, cross-validated with whole goals (scenarios) held out. Details, figures
and methods are in the explainer (see top).
- **Behavior**: every step stays within the target, and the last step tracks it (log-log slope 0.93, ρ 0.89). But
  for targets of a year or more, step 1 is about a month whatever the target, and steps 2–4 stay within a year.
  Without a target, the median last step is about 5 weeks.
- **Turn 1** (before the outline): the target is linearly readable (R² 0.98) and ordered along PC1 (|ρ| 0.88).
- **Later turns**: the target is still readable by a probe refit at that turn (R² 0.94–0.97), but the turn-1 probe
  transfers poorly (R² 0.56–0.73): the direction changes.
- **Next step**: before a step is written, the boundary activations add a small amount over a text-only baseline for
  that step's horizon (R² 0.86 → 0.89), above a shuffled-label control. Without a target: inconclusive.
- **Turn 1 and later steps**: turn-1 activations add ΔR² 0.08–0.11 for steps 1–4 over target + wording (shuffled
  control 0.00–0.05, one shuffle); the goal text may explain it.
- **No target**: a probe trained on plans with a target orders the self-chosen step horizons at ρ 0.78–0.86 from the
  boundary window, but is badly scaled (R² ≤ 0.52).

## Caveats

- One model, one run, thinking off, 12 goals. Greedy plans with a target are deterministic per prompt, so the 288
  conversations are 96 goal × target cells in 3 wordings, not 288 independent samples.
- "Time horizon" is not always read as "from today": 64% of plans have non-decreasing step horizons.
- Step horizons are stereotyped (counting up; the last step equal to the target), so the text alone predicts much of
  them. Compare any activation result with a text baseline (the analysis does).
- Positions `H`, `V` and `E` sit inside or after the reply and see the written text; `V` sees the value itself.

## Acknowledgements and citation

SPAR fall 2026, project *Planning Temporal Manifolds*. Mentors: Ian Rios-Sialer, Shantanu Darveshi, Justin Shenk.
The protocol follows the project proposal's "Continue" protocol. Author of this experiment: Augusto Nicola.

If you use this data, please cite the preprint the project extends and link this dataset:

```bibtex
@misc{ptm_multiturn_planning_2026,
  title        = {Multi-turn planning: Qwen3-14B residual-stream activations at the turn boundaries of five-step plans},
  author       = {Augusto Nicola},
  year         = {2026},
  howpublished = {Hugging Face dataset, anicola/ptm-multiturn-planning-qwen3-14b},
  note         = {SPAR fall 2026, Planning Temporal Manifolds. Code: github.com/unrulyabstractions/spar-planning-temporal-manifolds, commit a13c941}
}
```
