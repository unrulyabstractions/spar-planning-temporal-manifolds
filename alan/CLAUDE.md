# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this directory is

Working directory for Alan's Fall 2026 SPAR (Supervised Program for Alignment Research) project,
**"Planning Temporal Manifolds."** It holds the three reference PDFs below plus a small Python
package `ptm/` (activation capture and analysis), `scripts/`, and `tests/`.

**Repositories.**
- This repo is canonical and Alan's own: remote `origin` = `https://github.com/CodeReclaimers/SPAR-2026`
  (private, created 2026-09-25). All development, logs, and history live here.
- SPAR shared repo: `https://github.com/unrulyabstractions/spar-planning-temporal-manifolds`, local
  clone at `~/spar-planning-temporal-manifolds` (no commits as of 2026-09-25; contains an empty `alan/`
  folder). Alan occasionally copies **snapshots** of code, data, and results into `alan/` there. Treat
  it as a publication target, not a working tree: never develop in it, never push to it, and never copy
  anything into it unless Alan asks for a snapshot; when he does, copy from a committed state of this
  repo and record what was copied (commit hash, paths) in the progress log.

## Commands

Everything runs from the project `.venv` (created with `uv sync --extra dev`; torch 2.11+cu128,
transformers 5.17). Prefix GPU runs with `HF_HUB_OFFLINE=1` (weights are cached) and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

```
uv sync --extra dev                                    # environment
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest -q -s tests   # 20 tests, tokenizer-only, no GPU
.venv/bin/python scripts/gen_prompts.py data/prompts/investment_n2000_s0.parquet --n 2000 --seed 0
.venv/bin/python scripts/verify_capture.py --model Qwen/Qwen3-8B --n 4      # hooks vs library hidden states (must print VERIFY OK)
.venv/bin/python -u scripts/capture.py data/prompts/investment_n2000_s0.parquet runs/RUN --model Qwen/Qwen3-14B --batch-size 8 --split 19
.venv/bin/python scripts/analyze.py runs/RUN figures/RUN [--layers 0,10,20] [--positions T0,T8,R3] [--supervised]
.venv/bin/python scripts/gen_matrix.py data/prompts/matrix --seed 0    # matched matrix (scenario × horizon × rendering) + control sets
scripts/analyze_run.sh runs/RUN                        # standard post-capture bundle: sweep, path geometry, hypothesis checks, 3-D views
.venv/bin/python scripts/plot3d.py runs/RUN figures/RUN --cells 22:R0,37:T3 [--basis sample]   # interactive HTML (open in a browser)
```

Layout of `ptm/`: `horizons.py` (Horizon values, 17-point grid), `prompts.py` (parametric
intertemporal-choice prompts, section-marker template, all parameters as columns), `chat.py`
(chat template, transition-window positions, choice readout), `capture.py` (greedy generation,
then an unpadded per-sample forward with residual-stream hooks), `store.py` (bf16 safetensors
shards + parquet index; lazy per-cell reads), `analysis.py` (per-cell PCA, Spearman with log
horizon, reward-ratio control, choice silhouette, figures).

**Capture facts established 2026-09-19 (do not re-derive):**
- Stored `layer` axis: 0 = embedding output; `l` = residual stream after decoder layer `l-1`
  (the prior work's `resid_post` of layer `l-1`); no final norm applied. transformers'
  `hidden_states[-1]` is post-norm and differs from the raw last-layer residual by ~90% relative L2.
- Unpadded single-sequence forward passes reproduce the library's `output_hidden_states` bit-exactly
  at every layer. Padded batched passes differ by 0.6% (layer 1) growing to 8% (layer 35) relative
  L2 in bf16. Capture therefore generates in batches but runs the teacher-forced pass one sequence
  at a time. Reported activations are exact for the generated text; batching can still change which
  text is generated on near-tied greedy steps.
- bf16 logits at |logit|≈60 have a 0.5 ulp, so the two label logits are computed in float32 from the
  normed residual (`logit_a`, `logit_b`); `p_short` is the pairwise sigmoid of their difference.
- Qwen3-14B fits only with an explicit device map: embeddings + 19 decoder layers on GPU 0, the
  remaining 21 + norm + lm_head on GPU 1 (`--split 19`); peak 13.4 / 14.6 GiB at batch 8, 3.6 s per
  batch of 8 with 48 new tokens. accelerate's automatic split offloads to disk and must not be used.
- Qwen3 no-thinking transition window is 9 tokens: `<|im_end|>`, `\n`, `<|im_start|>`, `assistant`,
  `\n`, `<think>`, `\n\n`, `</think>`, `\n\n` (positions T0–T8); response positions R0..R7. The
  choice token sits at R3 when the format is followed (`I`, ` choose`, `:`, ` a`/` b`).

## The three reference documents

- `Planning Temporal Manifolds 2.pdf` (47 slides) — the kickoff deck walked through on 2026-09-14.
  **Authoritative for the phase plan, deadlines, logistics, and the candidate research directions.**
  Slides 33–47 are per-week title cards and per-mentee update templates.
- `Planning Temporal Manifolds 1.pdf` (8 pages) — the original SPAR project proposal. Its motivation,
  notation, risk analysis, and backup plans still apply. Its phase table and its specific
  multi-turn protocol are **superseded by the deck**: at kickoff the mentors framed weeks 1–4 as
  open exploration in which each mentee chooses a direction, and did not present the proposal's
  protocol or RQ1–RQ3 as fixed.
- `preprint.pdf` (28 pages) — *Temporal Concepts and their Shape in Large Language Models*
  (Rios-Sialer, Darveshi, Jiang, Paudel, Pronina, Bandyopadhyay, Shenk; arXiv:2606.05194).
  The prior work this project extends. Mentors are Ian Rios-Sialer, Justin Shenk, and Shantanu
  Darveshi, all co-authors.

**Prior code.** The only existing code is the public repo `unrulyabstractions/temporal-awareness`
(GitHub and Hugging Face; last push 2026-03-21). There is no private mentor repo. The public repo
has a multi-backend `ModelRunner` (TransformerLens / nnsight / pyvene / raw HF hooks), a `TimeValue`
class with unit parsing, a grid-based intertemporal prompt generator with formatting variations, and
token-position utilities, with heavy dependencies (`transformer_lens`, `nnsight`, `pyvene`, and a git
dependency on `justinshenk/latents`). Treat it as reference material to read and selectively
borrow from, not a base to inherit; whether to fork it or start fresh is an open decision.

Read PDFs with the Read tool's `pages` parameter (max 20 pages per call).

## Framing from the kickoff deck

**Motivation.** Planning horizon shapes behavior: the same "create an app" request yields quick
functional code under an hours-long horizon and production-grade code under a months-long one.
The internal horizon is not directly observable, chain of thought can be unfaithful to it, and a
long hidden horizon is the substrate for scheming. **Goal:** monitor (and eventually control) the
internal time horizon by inferring it from activations.

**Approach.** Mechanistic interpretability plus representation engineering, in three linked pieces:
parametric querying → geometry as a function of time; behavioral modeling → behavior as a function
of time; then behavior as a function of geometry via geometric intervention. The concept is to be
characterized on three axes: **representation, causal, functional** ("good structure" — does the
geometry have the shape needed to support the computation, not merely correlate with the label).

**Open questions the mentors named.**
- Stability and generalizability: is the manifold the same across tasks and contexts (e.g. at the
  end-of-turn token)? Can it predict the planning horizon of an arbitrary generation?
- Functionality: how much does horizon affect behavior? Can the manifold be used to steer?

**Candidate directions offered (each mentee picks one, or proposes another).**
1. **Characterizing activation-space structure.** What kind of concept is time horizon? If not a
   linear direction, how is it operated on — geometric, topological, spectral? What are all the
   ways a signal can be encoded in R^n? How does the visualized geometry deform as prompts move
   from highly formatted / decomposable / explicit toward ecologically valid / implicit? A
   mentor finding (Shantanu, Aug 2026) is cited: at layer 21, at the newline position, each time
   unit forms a distinct manifold, and overlaid equivalent expressions line up. Prompt variants
   preserve "structure" but not absolute geometry. Reference named on the slide: *From Directions
   to Regions: Decomposing Activations in Language Models via Local Geometry*.
2. **Mapping behavior to geometry.** Use "toyfied" parametric scenarios — a real LLM on a highly
   constrained task — with explicit numerical behavior models. References named: *Manifold Steering
   Reveals the Shared Geometry of Neural Network Representation and Behavior*; *Transformers
   Represent Belief State Geometry in their Residual Stream*.
3. **Circuit tracing.** In a toyfied intertemporal-choice scenario, map how the model goes from
   reading the options and horizon, through number comparison, to emitting the label — "path
   patching on steroids." Caveat on the slide: computation is often distributed, sparse, and
   redundant, which makes this hard.
4. Other.

**Out of scope** (carried from the proposal and not contradicted at kickoff): frontier/closed
models, training-time interventions, agentic tool use. If steering is reached, report it as effect
sizes and layer locations, never as released vectors or optimization recipes.

## Phase plan and deadlines (from the deck)

| Phase | Weeks | Dates | Goal |
|---|---|---|---|
| Exploration | 1–4 | Sep 14 – Oct 11 | Formulate falsifiable hypotheses |
| Understanding | 5–8 | Oct 12 – Nov 9 | Test the hypothesis |
| Distillation | 9–12 | Nov 9 – Dec 7 | Polish, write, measure uncertainty |
| Communication | 13–14 | Dec 7 – Dec 19 | Presentation |
| Paper writing (optional) | 14–16 | after Dec 14 | ICML 2027, January submission |

Week-by-week intent from the title cards:

| Week | Dates | Intent |
|---|---|---|
| 1 | Sep 14–20 | Kickoff, logistics, background |
| 2 | Sep 21–27 | Playing around, literature, visualizations, replications |
| 3 | Sep 28 – Oct 4 | Define scope, sketch narrative |
| 4 | Oct 5–11 | Formulate hypothesis + planned support/evidence |
| 5 | Oct 12–18 | Baselines + first big experiment |
| 6 | Oct 19–25 | Pivot or double down? |
| 7 | Oct 26 – Nov 2 | Iterating, debugging, expanding |
| 8 | Nov 3–9 | What is our main result + contribution? |
| 9 | Nov 9–15 | Need a late pivot? |
| 10 | Nov 16–22 | Scale up: more models, tasks, samples? |
| 11 | Nov 23–29 | Uncertainty, counter-evidence (or characterizing a negative result) |
| 12 | Nov 30 – Dec 7 | Cleaning code, writing (blog?) |
| 13 | Dec 7–13 | Final report, poster, talk |
| 14 | Dec 14–19 | Demo day prep |

| Deadline | Date |
|---|---|
| Midterm outline and instructions released | Oct 12, 2026 |
| **Midterm report** (and feedback form) | **Nov 2, 2026**, end of day AoE |
| Final report outline and instructions released | Nov 24, 2026 |
| **Poster, lightning talk, final report** (and feedback form) | **Dec 14, 2026**, end of day AoE |
| SPAR Demo Day | Dec 19, 2026 |

Follow the phase in effect. During Exploration (weeks 1–4) the deliverable is a chosen direction
and a falsifiable hypothesis with planned evidence, not a pipeline. Do not build infrastructure
that presumes the proposal's protocol unless Alan has chosen that direction. Ian's advice: get on
writing early; sketching the midterm and final report helps prioritize.

## Logistics

- Weekly team meeting. **Fill in the personal update slide (Update / What's Next / Discussion-Blockers)
  at least 24 hours before it.** Keep it short; put figures on extra slides.
- Weekly 1-1 with Ian (scheduled by DM).
- Compute: **Vast.ai** is the program-provided compute. Local GPUs (below) are for prototyping.
- Tooling named at kickoff: GitHub, Vast.ai, OpenReview, Claude Code.
- Ian recommends keeping a personal Google Slide deck for notes, figures, brainstorming, and review.
- Lean toward over-communicating; reach mentors on Slack.
- Team: mentors Ian Rios-Sialer (San Francisco; London in October), Justin Shenk (Berlin), Shantanu
  Darveshi (Mumbai). Mentees: Alan, Augusto "Gus" (Buenos Aires), Jared (New York), Soham.

## Notation and design constraints (from the proposal; retained as conventions)

Use these terms verbatim in code, docs, and logs whenever the concept applies.

- `H_target` — the horizon assigned to the whole task in the prompt ("2 weeks", "20 years").
  We set it, so it is a clean label.
- `H_step` — the horizon of an individual plan step, read from the model's own output. Treat as a
  **weak, noisy label**; always cross-check against `H_target` and report both. Its semantics
  (step duration vs. reach into the future vs. deadline) are undefined in the proposal and must be
  fixed explicitly before any parser is written.
- `L` — model depth; report capture depths as fractions of `L` (the proposal used
  `ℓ ∈ {0.4L, 0.6L, 0.8L}`). The prior work found fractional depth, not absolute layer index, is
  the coordinate that compares across models; peak causal attention sits at 0.61–0.69 of depth in
  every family.
- **Boundary tokens** — change-of-turn sequences (`<|im_end|>`, `<|im_start|>assistant`) and
  thinking delimiters (`<think>`, `</think>`). If capturing at these positions, never capture
  tokens inside the chain of thought.
- **Chat-template tokens are verified per model family and generation, never assumed.** Each
  family has its own turn tokens (Qwen: `<|im_end|>`/`<|im_start|>`; Llama: `<|eot_id|>`,
  `<|start_header_id|>`, `<|end_header_id|>`; Gemma: `<end_of_turn>`/`<start_of_turn>`/`model`;
  Mistral: `[/INST]` only). Write a test that asserts the captured positions are the intended
  tokens before trusting any downstream number. The preprint's target `Qwen3-4B-Instruct-2507` has
  no think block; the cached Qwen3-8B/14B/32B are hybrid models that do; Qwen3.5 is a different
  generation whose template has not been verified for this project.

**The proposal's protocol (one candidate instantiation, not binding).** Prompt gives scenario, task,
and `H_target`, ecologically phrased. Model returns a high-level plan; each "Continue" turn elicits
one step in a fixed `Step / Time horizon / details` format, ending with "Plan Completed". Main runs
no-thinking via a pre-filled empty think block; thinking mode only to validate code paths. Captured
activations keyed by `(prompt_id, model_id, sample_uid, token_position, layer_depth, step_index)`.
Analysis by PCA colored by `H_target`, `H_step`, `prompt_id`; ordinality as rank correlation between
PC1 and stated horizon, fit per layer and per token (the preprint's Table A.3 reports |ρ| 0.79–0.97).

## The main scientific risk: confounded geometry

Apparent horizon structure may be driven by prompt-surface features (the literal horizon phrase,
task keywords, a fixed response format) rather than an internal representation. This applies to
every candidate direction, not only the planning protocol. Any design therefore needs **controls,
not a single prompt template**: paraphrase and format randomization, and content-matched horizons.
If structure survives only when the literal horizon token is present, report that as a scoped
negative result. Analysis code should make it trivial to run the control condition alongside the
main one; do not build a pipeline where the control is an afterthought. The preprint's own
specificity control was a reward sweep in the same prompts; a design without a reward field needs a
substitute graded non-temporal quantity or the specificity argument is lost.

Other contingencies named in the proposal: weak signal at the chosen positions → widen capture to
other tokens and move from unsupervised PCA to supervised linear probes; poor format adherence →
few-shot exemplars or constrained decoding, with adherence rates logged per model; compute-limited →
fewer families, smaller target model, fewer horizon values.

## What the prior work established (constraints on this project's assumptions)

Target model in the preprint was `Qwen3-4B-Instruct-2507`; families studied were Qwen3, Llama-3.1-8B,
Gemma-2-9B, Mistral-7B, each on a different decision domain (investment, health, climate,
education; the target model also ran on a startup domain). Findings this project builds on:

- Time horizon (seconds to centuries) is an **ordinal, non-linear** gradient at turn-transition
  tokens, at mid-to-upper layers. The direction carrying it moves from token to token and layer
  to layer, so **do not assume a single global direction**; fit per layer and per token. The
  ordering also appears at the first response tokens at the same strength.
- The same activations encode reward (swept on a log grid) more weakly than horizon, which is the
  control that makes the ordering horizon-specific rather than "any graded quantity."
- Decodability (linear probe), causal effect (activation patching), and writability (steering)
  pick out **different depths**. A probe alone does not locate where a preference is used.
- Representation does not deliver behavior: every family carries the geometry, only Gemma-2-9B
  reasons coherently over it. Small models often do not use the representation (low temporal
  reasoning, unstable preference), which is why the deck's "mapping behavior to geometry"
  direction leans on toyfied tasks with explicit behavior models.
- Contrastive steering works only inside a bounded coefficient range; beyond it, coherence drops
  because the linear vector leaves the locally linear region of a curved manifold.
- Their prompt template held section markers (SITUATION / TASK / OBJECTIVE / CONSTRAINT / ACTION /
  FORMAT) at fixed token positions across variants; the horizon line was a single optional
  sentence, and prompts with it null formed their own cluster. Their limitations section warns that
  task vocabulary entangles with horizon and cites evidence that linear representations change
  dramatically over a conversation.

## Compute

**Vast.ai is the program-provided compute** (per the kickoff deck); account and instance details are
not yet recorded here. Use it for anything that needs more than the local machine, and record the
instance type, image, and cost per run in the progress log.

### Local hardware (scanned 2026-09-13), for prototyping

| Resource | Value |
|---|---|
| GPUs | 2 × NVIDIA RTX 4080 SUPER, 16 GB each (32 GB total), compute capability 8.9 |
| GPU interconnect | PCIe via host bridge (PHB), no NVLink |
| Driver / CUDA | driver 580.173.02, CUDA 13.0 driver API; system torch is 2.10.0+cu128 |
| CPU | AMD Ryzen 9 9950X, 16 cores / 32 threads, AVX-512 present |
| RAM | 186 GB (swap is 8 GB and was ~94% used at scan time; do not count on it) |
| Disk | 3.6 TB NVMe, ~387 GB free; the Hugging Face cache already holds ~300 GB |
| System Python | 3.12.3 with torch, transformers 5.2.0, scikit-learn; **no** nnsight, transformer_lens, accelerate, bitsandbytes, or vllm |
| Package manager | `uv` is installed; give this project its own `.venv` rather than using system Python |

What fits locally for hook-based capture in `transformers` (32 GB VRAM is the binding constraint):

| Model | bf16 weights | Fits in 32 GB VRAM? |
|---|---|---|
| Qwen3-8B | ~16 GB | Yes, with headroom |
| Qwen3-14B | ~28 GB | Yes, sharded across both GPUs; ~2 GB/GPU left for KV cache and captured states, so keep contexts modest |
| Qwen3-32B (dense) | ~64 GB | No. Needs 4-bit quantization (~18–20 GB) or CPU offload to RAM (fits, but each forward pass streams offloaded layers over PCIe, so multi-turn generation is slow) |

Quantization is a design decision to raise with the mentors, not a silent default: the preprint
kept 4-bit variants as separate rows from full-precision models. Ollama can run 27–35B quantized
models here (several are already pulled) and is useful for prompt/format prototyping, but it
serves GGUF weights with no residual-stream access, so it cannot be the capture path.
TransformerLens converts weights and roughly doubles memory, and its model whitelist lags newer
Qwen checkpoints; raw Hugging Face hooks with `device_map="auto"` are the realistic path at 14B+.

Weights already in the Hugging Face cache and relevant here: Qwen3-8B, Qwen3-14B, Qwen3-32B,
QwQ-32B, Qwen3.5-{0.8B, 2B, 4B, 9B}, Mistral-7B-Instruct-v0.2. **Not** cached: the preprint's
target `Qwen3-4B-Instruct-2507`, Llama-3.1-8B-Instruct, gemma-2-9b-it.

## Working conventions specific to this project

- This is a research project with quantitative deliverables. The rules in the global CLAUDE.md on
  reproducible numbers, gating each pipeline stage on an audit of its output, and describing
  methods from the code rather than from memory apply with full force. Any parser that reads a
  horizon out of model text is exactly the kind of heuristic that silently carries corpus
  assumptions: hand-audit a sample per model before any downstream count depends on it.
- Progress logs follow the global default (`progress-YYYYMMDD.md` in this directory) until a
  repo-level override says otherwise.
- Weekly cadence: the update slide is due 24 hours before the team meeting. When asked to help
  prepare it, draw on the progress logs, not recollection.
- The proposal's application questions (page 7) remain a useful checklist for experimental design
  review: what distinguishes "no signal" from "confounded signal" from "signal elsewhere"; what
  control separates literal-wording from internal-representation explanations; when a method
  other than PCA is the better choice.
