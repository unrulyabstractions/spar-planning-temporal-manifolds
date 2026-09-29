# Plan: Robustness of the time-horizon representation in a mid-size instruct model

Status: v1. Decisions from Jared (2026-09-21): one model for now; thinking ON as the
primary condition with an OFF toggle. Nothing on the 8B model has been run yet.
Date: 2026-09-21

## 0. Where this comes from

`spar_starter_geometry.py` is a 300-line distillation of the intertemporal
geometry pipeline in `temporal-awareness/` (Rios-Sialer et al. 2026,
"temporal preference" in Qwen3-4B-Instruct-2507):

| Starter piece | Prior-project source |
|---|---|
| Prompt template ("greatest benefit for this time horizon") | `src/intertemporal/formatting/configs/default_prompt_format.py:61` |
| Log-swept horizons, PCA per (layer, token), Spearman vs log horizon | `src/intertemporal/geometry/`, `scripts/intertemporal/compute_geometry_analysis.py` |
| Turn-suffix + response-token positions | `src/intertemporal/common/semantic_positions.py` |
| Temporal reasoning / order / label stability | `scripts/intertemporal/coherent_behavior.py` ("instrumental incoherence" in the paper) |

The prior paper found the horizon signal concentrated in a mid-to-late
subgraph (L21/L24 attention in Qwen3-4B) and steerable with CAA at L19-22.
The starter on Qwen3.5-0.8B reproduces the geometry (peak |rho| 0.955) but
the behavior tests show pure position bias: the tiny model never reads the
content. The new work asks whether the horizon representation in a model
that *does* read the content is robust, and whether it is actually used.

## 1. Research question

**Is the linearly-readable time-horizon representation a stable, prompt-
independent quantity, and does the model's choice causally depend on it?**

"Robust" is operationalized as five sub-questions, each with a pass metric:

| # | Sub-question | Manipulation | Metric | Pass |
|---|---|---|---|---|
| R1 | Survives rewording | 5 paraphrases of the constraint sentence, 3 of the framing | Horizon probe trained on template A, tested on B: Spearman rho and R² | rho > 0.8 cross-template |
| R2 | Survives domain change | Investment vs. 3 other scenarios (medical treatment, infrastructure project, personal goal) | Same cross-domain transfer | rho > 0.7 |
| R3 | Encodes duration, not surface unit | Same duration in different units ("12 months" / "1 year" / "365 days") | Within-duration spread vs. between-duration spread along the horizon axis | Unit variance < 20% of horizon variance |
| R4 | Invariant to nuisance factors | Swap option order, relabel a/b to 1/2, change rewards | Recolor scatter by nuisance factor; rho of PC1 with reward must stay near 0 | Horizon rho stable within 0.05; reward rho < 0.2 |
| R5 | Causally used | Add ±alpha * horizon direction at the peak layer on the turn tokens | Choice flip rate vs. alpha; random-direction control | Monotone flip curve, control flat |

R1-R4 are representational. R5 is the "used, not just encoded" test and is
the one that turns this from descriptive into a finding.

## 2. Model choice

Target hardware is the existing vast.ai setup: one RTX 4090, 24 GB, ~$0.50/h.

| Candidate | bf16 weights | Fits 4090? | Gated? | Notes |
|---|---|---|---|---|
| Llama-3.1-8B-Instruct | 16 GB | Yes | Yes (license click + token) | Already in the prior registry with results to compare against. Plain chat template, no think block. Easiest. |
| Gemma-2-9B-it | 18 GB | Yes, tight | Yes | Needs eager attention (softcapping), slower. Gemma Scope SAEs exist if we want them later. |
| Qwen3-14B | 28 GB | No; 8-bit (~15 GB) or a 48 GB card | No | Same family as the paper's model, best comparability. Has thinking mode to disable. |
| Qwen3.5-27B 4-bit | ~15 GB | Yes | No | Hybrid linear-attention architecture (gated delta net): per-layer hidden states are not a plain residual stream and hook/patching tooling does not support it. 4-bit also adds quantization noise to the geometry. Avoid for interp. |

**Decision: Qwen3-8B** (`Qwen/Qwen3-8B`, not on the original shortlist).
The thinking-on requirement rules out Llama-3.1-8B-Instruct and Gemma-2-9B,
which have no native think block; "thinking" there would mean a
chain-of-thought instruction that changes the prompt itself. Qwen3-8B has
the native `enable_thinking` toggle, fits a 4090 in bf16 (16 GB), is
ungated, is in the prior project's registry, and is the same family as the
paper's Qwen3-4B. Qwen3-14B is the same family but needs 8-bit on a 4090.
The notebook's `MODEL` knob makes switching a one-line change.

Estimated compute: the starter bank is 116 prompts. Every experiment below
is at most ~2,000 forward passes on an 8B model, so under an hour of 4090
time per experiment and under 10 GPU-hours total.

## 3. Architecture

Extend the standalone starter rather than the prior repo. The prior repo
is heavy (uv, TransformerLens, 40+ modules) and its prompt/position
machinery is tied to Qwen3-4B. We borrow its *ideas* (prompt configs,
semantic positions, CAA steering) but keep the new code small.

```
spar_horizon/
  prompts.py        # templates, domains, units, nuisance factors -> prompt bank
  model_io.py       # load model, chat-template, generate, hidden states
  positions.py      # turn-suffix length from the template, not hardcoded
  geometry.py       # PCA / Spearman sweep, horizon probe (ridge on log horizon)
  behavior.py       # parse choice, three coherence tests
  steering.py       # add/ablate direction via forward hook
  run_*.py          # one entry point per experiment R1..R5
tests/              # unit tests on the pure-python parts (prompt bank, parser, positions)
```

Key change from the starter: `N_POSITIONS = 9` is Qwen3.5-specific. The
turn-suffix length must be computed by tokenizing the template with and
without `add_generation_prompt` and taking the difference.

Key addition: a *horizon probe* (ridge regression from residual to log
horizon) alongside PC1. PC1 only works when horizon is the dominant
variance; the probe lets us measure transfer across templates (R1, R2) and
gives the steering direction for R5.

## 4. Task breakdown (each task is one PR-sized unit with its own check)

### Phase 0: Infrastructure (blocked on HF token + vast.ai access)
- 0.1 Port `load_model` / chat-template call to be model-agnostic (Llama has no `enable_thinking`). Check: tokenizes a prompt without error for both Qwen and Llama tokenizers.
- 0.2 Compute turn-suffix length from the tokenizer. Check: equals 9 for Qwen3.5-0.8B, prints the value for Llama.
- 0.3 Rent 4090, pull model, run the unmodified starter bank. Check: table prints, PNG written, behavior lines print.

### Phase 1: Baseline on the target model
- 1.1 Run geometry sweep + behavior on the starter bank. Check: peak |rho| and display layer recorded; behavior tests no longer show position bias (order stability well above 0).
- 1.2 Add the ridge horizon probe per layer with 5-fold CV. Check: R² per layer table; best layer agrees with the PCA peak within a few layers.
- 1.3 Save activations to disk (npz) so R1-R4 do not re-run the model for the baseline bank.

### Phase 2: Robustness (R1-R4, all representational)
- 2.1 Prompt bank generator with templates × domains × units × nuisance. Unit-tested. Check: expected counts, no duplicate prompts, every prompt parses back to its metadata.
- 2.2 R1 paraphrase: extract, train probe on each template, test on the others. Output: transfer matrix heatmap.
- 2.3 R2 domain: same, across domains.
- 2.4 R3 units: for 6 durations × 3 unit spellings, project onto the horizon axis and compare within/between spread.
- 2.5 R4 nuisance: rerun swap / relabel / reward variants, recolor scatter, report horizon rho and reward rho side by side.

### Phase 3: Causal use (R5)
- 3.1 Forward hook that adds alpha * d at layer L on the last k prompt tokens. Check: alpha = 0 reproduces the unhooked logits exactly.
- 3.2 Sweep alpha on the 27 "only the near option delivers" prompts and their swapped versions. Report flip rate vs alpha, with random-direction control at matched norm.
- 3.3 Ablation: project out d at high-horizon prompts; does the choice move toward the short-horizon answer?

### Phase 4: Write-up
- 4.1 One results markdown with the five pass/fail rows filled in.
- 4.2 Figures: transfer matrices, unit-spread bar chart, steering curve.

## 5. Decisions and remaining questions

Decided (2026-09-21):
- One model for now: Qwen3-8B.
- Thinking ON is the primary condition; every experiment also runs with
  thinking OFF via the `THINKING` knob, so each result table has two rows.
- Standalone code in `jared/`; `horizon_geometry.py` is
  the source of truth and `build_notebook.py` regenerates the `.ipynb`.

Thinking ON changes the extraction, and this is handled in the port:
- The turn suffix ends with an open `<think>\n` instead of an empty block,
  so it is shorter (7 tokens on Qwen3.5 vs 9). Positions come from the
  tokenizer, not a constant.
- The answer tokens are the first `N_RESPONSE` tokens after `</think>`.
  Prompts whose think block does not close within `MAX_THINK_TOKENS` are
  skipped and counted.
- Generation is far more expensive per prompt. The 116-prompt bank at
  512 think tokens is still minutes on a 4090.

Still open:
1. Does "robustness" mean R1-R5, or robustness *under repetition* (the
   patience-degradation link in `subsection_temporal_stability.tex`)?
   Assumed R1-R5 until told otherwise.
2. Budget cap for vast.ai. Assumed under 10 GPU-hours.
