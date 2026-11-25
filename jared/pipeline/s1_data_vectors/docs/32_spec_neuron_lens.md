# Spec: horizon and decision neurons against the Neuron Lens catalog (stage A, local)

Approved by Jared 2026-11-12. Stage A only; stage B (exact activations on the box) is
proposed after A's results.

## Question

Which Qwen3-8B MLP neurons track the stated horizon and the near-vs-far decision at the
cells our findings rest on, and are their Neuron Lens labels about time or choice more
often than chance?

Neuron Lens (https://neuron-lens.com, catalog `stdlogits/neuron-lens` on Hugging Face,
CC BY 4.0, by a9lim and treetown with Claude) labels both poles (signs) of every MLP
neuron of Qwen3-8B (36 x 12,288) with Claude Haiku 5.5: `fires` (where it fires) and
`writes` (what its write pushes), each with a confidence.

## Cells

| cell | layer | prompts |
|---|---|---|
| suffix newline (after `<|im_end|>`) | 16 | all with a horizon (471) |
| pre-label `:` | 22, 24 | natural closes only (forced closes confound every cell after `</think>`) |

Layer convention: our `hidden_states[L]` is the input to decoder block L, which Neuron
Lens calls "L{L}". Block L's neurons read it (after block L's attention); block L-1's
neurons wrote into it.

## Targets

- **Horizon:** log10 horizon in years. Ranked *raw* and *within unit* (target and
  activation each centred on its per-unit-word mean, as `probes.within_unit_r2` does).
  The raw list is expected to be unit-word neurons; the within-unit list tests magnitude.
- **Decision:** near log-odds at the pre-label token, `e0_runs.saved_decision` (rows
  with NaN log-odds dropped).

## A1. Approximate neuron activations (read side)

Per prompt, block L's neuron activations `silu(W_gate u) * (W_up u)` with
`u = RMSNorm_post(h)`, `h = hidden_states[L]` at the cell. Pole + is the activation,
pole - its negation. Rank poles by Spearman rho with each target.

Caveat: the MLP's real input is `h + attn_L(h, context)`; block L's attention output
was not saved, so these activations are approximate. Stage B measures the error.

## A2. Write side (weights only)

Cosine of every block L-1 pole's write (`±W_down[:, n]`) with our directions at the
cell: the least-squares direction of the within-unit horizon target, and of the near
log-odds, each fitted on the cell's activations (ridge, penalty by CV).

## Significance

Permute each target 200 times (within unit for the within-unit ranking) and take the
95th percentile of the largest |rho| over all poles of the cell: the family-wise
threshold. Report how many poles pass it.

## Label check

For the top 20 poles per (cell, target, ranking) by rho (A1) and by cosine (A2):
their `fires` / `writes` labels and confidences. Enrichment: the share of top poles
whose `fires` or `writes` label matches a keyword family, against the share among all
poles of that layer (Fisher exact test, one-sided). Both families are frozen here,
before any result is seen (case-insensitive, whole words):

- **time:** time, times, timing, duration, durations, period, periods, horizon,
  horizons, year, years, yearly, annual, month, months, monthly, week, weeks, weekly,
  day, days, daily, hour, hours, hourly, minute, minutes, decade, decades, century,
  centuries, millennium, future, later, soon, long-term, short-term, wait, waiting,
  delay, delays, delayed, deadline, temporal, eventually, tomorrow
- **choice:** choice, choices, choose, chooses, chose, option, options, decision,
  decisions, decide, prefer, preference, select, selection, alternative, alternatives,
  reward, rewards, payout, payouts

Ambiguous words (second, past, now, since, age) are left out.

## Data and outputs

- Labels for blocks 15, 16, 21-24 (`qwen3-8b/neurons/labels/L*.parquet`) and the
  needed tensors of `Qwen/Qwen3-8B` (gate, up and post-attention norm of blocks 16, 22,
  24; down of 15, 21, 23; about 0.9 GB, HTTP range reads of the safetensors shards),
  fetched by `analysis/neuron_lens_fetch.py` into `$RESULTS/neuron_lens/`, with the
  catalog and model commit hashes.
- `analysis/neuron_lens_compare.py` -> `$RESULTS/analysis/neuron_lens_*` (json, csv,
  png) and a results doc.
- Test: on the tiny random Qwen3 of `tests/`, the A1 formula equals the hooked MLP
  activation exactly when given the true MLP input.

## Stage B decision

If A1 has poles past the threshold or the labels show enrichment: propose a
forward-only rerun of the cached traces on the 4090 hooking `act_fn(gate) * up` at the
kept positions (about $0.50-1, budget from Jared), which also measures A1's error.
Otherwise report A and let Jared decide.

Out of scope: attention heads, think positions, Gemma.
