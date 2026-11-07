# prompt_bank

The prompts for the Planning Temporal Manifolds experiments, the tools to check them, a baseline run of the
models on them, and a web page for reviewing all of it by hand.

The design is the one in the midterm report (`midterm_report/latex/report.tex` on the `midterm-report` branch),
Methods and Appendix A.1. This directory implements tier **T0** (structured prompts). The realistic tiers T1–T3
are deliberately not built yet: the report's roadmap (A.6, step 2) builds realistic data only after steering
passes its gate on T0.

## The three steps

```
uv sync --all-extras                                   # once: creates .venv (torch, transformers, flask, ...)

# 1. Build the bank and have a second model check every prompt
.venv/bin/python generate.py --out data/t0_s0                                   # checker: local Qwen3.5-27B
.venv/bin/python generate.py --out data/t0_s0 --checker anthropic:claude-haiku-5-5            # Anthropic API
.venv/bin/python generate.py --out data/t0_s0 --checker openrouter:anthropic/claude-haiku-5.5  # OpenRouter
.venv/bin/python generate.py --out data/t0_s0 --checker none                    # build only: no GPU, no API

# 2. Run a model, unsteered, on every prompt (thinking on, greedy, 8,192-token cap)
.venv/bin/python -u baseline.py --bank data/t0_s0                               # Gemma 4 31B-it
.venv/bin/python -u baseline.py --bank data/t0_s0 --model Qwen/Qwen3-32B --batch-size 8

# 3. Review prompts, checker reports and responses in the browser
.venv/bin/python review.py --bank data/t0_s0        # open http://127.0.0.1:5000
```

Every script has `--help`. Each one writes its outputs next to its inputs and can be rerun safely: building is
deterministic and a no-op when the prompts already exist, and checks and baseline runs resume where they stopped.

**Where to run what.** Building and reviewing need only a laptop. A local checker or baseline needs a GPU that
holds the model in bf16: Gemma 4 31B-it is 62.6 GB of weights and Qwen3.5-27B is 55.6 GB, so both need an
A100/H100 80 GB (on Vast.ai). The 2026-10-17 pilot measured Gemma at about 13 s per prompt at batch 4 on one A100,
so the 700 prompts take about 2.5 GPU-hours.

**Hosted checkers** run either through the Anthropic API (`anthropic:<id>`, Claude models only, ids such as
`claude-haiku-5-5`; key `ANTHROPIC_API_KEY`) or through [OpenRouter](https://openrouter.ai) (`openrouter:<id>`,
any provider, ids such as `anthropic/claude-haiku-5.5` or `openai/gpt-5.6-luna` from
<https://openrouter.ai/api/v1/models>; key `OPENROUTER_API_KEY`). Keys are taken from the environment, or else
from a `.env` file (`NAME=value` lines) in `prompt_bank/` or the repository root. Both locations are git-ignored;
never commit a key. Checking all 700 prompts with Haiku 5.5 costs well under a dollar.

## What is in the bank

`data/t0_s0/prompts.jsonl` holds 700 prompts in two splits. Every prompt uses this template (the report's sample):

```
SITUATION: Plan for the future of the household.
TASK: You, the head of the household, are tasked to choose the best investment:
a) 10,000 dollars in 1 year.
b) 50,000 dollars in 10 years.
OBJECTIVE: Think deeply about which option is preferable.
CONSTRAINT: You must select the option that provides the greatest benefit for this time horizon: 8 years.
ACTION: Select one of the two options. Provide reasoning.
FORMAT: I choose: <a) or b)>. My reasoning: <1-3 sentences>
```

### The decision rule

An option **pays off within the horizon** when its delay is at most the horizon (a payout exactly at the end of
the horizon counts). The far reward is always larger (2–50 × the near reward) and the far delay always longer.

| regime | which options pay off | correct answer |
|---|---|---|
| `none` | neither | none: used for extraction, never scored |
| `near_only` | only the near option | the near option |
| `both` | both | the far option |
| `no_horizon` | (no CONSTRAINT line) | none: the null control |

Durations are compared exactly (`fractions.Fraction`, in days, with 1 year = 365.25 days, 1 month = 1/12 year,
1 week = 7 days), and always using the value *as written* in the prompt, since that is what the model reads.

### Extraction split (500 prompts): for fitting steering vectors

| parameter | values | drawn |
|---|---|---|
| horizon | 48 values, 30 seconds to 500 years, log-spaced (report Table 2) | balanced: each value 9 or 10 times |
| horizon rendering | natural unit; or a *unit rewrite*: the same duration as a whole number of another unit, within 2%, number ≤ 1,000 (e.g. "23 months" → "100 weeks") | uniform over the natural form and its rewrites |
| near option | $1k, 2.5k, 5k, 10k, 25k, 50k, 100k after 1 week, 1 month, 3 months, 6 months or 1 year | uniform |
| far option | near reward × 2, 5, 10, 25 or 50, after 2, 5, 10, 20, 30 or 50 years | uniform |
| option order | near first or far first | 50/50 |
| CONSTRAINT phrasing | the report's 4 phrasings (`pbank/t0.py`) | uniform |
| labels | `a)`/`b)` or `1)`/`2)` | uniform |
| null control | 29 prompts: copies of 29 horizon prompts with the CONSTRAINT line removed | random sources |

Seed 0 gives: 471 horizon prompts (279 `none`, 106 `near_only`, 86 `both`) and 29 `no_horizon`; 172 horizons
rewritten; 5 prompts whose horizon equals a delay exactly (flagged `horizon_equals_a_delay`). Most prompts have no
correct answer because half the grid is shorter than any near delay; the report intends this (A.1).

### Verification split (200 prompts): for scoring behaviour

100 items near the boundary where the answer flips, each in both option orders. For item *i* the ratio H/D
(horizon over far delay) is the *i*-th of 100 values log-spaced over [1/2.8, 2.8]; the far delay D is balanced over
the six far delays; H is written in years like D, to two decimals ("17.86 years"). The near delay is always
shorter than H, so every prompt is `near_only` (H < D) or `both` (H > D). No ratio is exactly 1, and the nearest
ones (0.990, 1.010) are far enough from it that rounding H can never create a tie with D.

### Choices made here that the report does not spell out

These are implementation decisions, recorded so that they can be questioned:

1. **Ties count as paying off** (delay ≤ horizon), as in the earlier project code. They occur only in the
   extraction split (5 prompts at seed 0), and are flagged.
2. **A unit rewrite is only used if it keeps the prompt's regime.** A 2% rewrite can cross a delay: "3 months"
   ties a 3-month delay, but "13 weeks" is 0.3% shorter. Such rewrites would change the correct answer.
3. **Rewrites are capped at 1,000** in the rewritten unit (so no "5040 seconds"), and only use the seven natural
   units (no quarters, decades or centuries).
4. **Horizons are balanced** across the 48 grid values, so per-horizon averages have equal support.
5. **Null prompts are twins** of horizon prompts (the report: "the same prompt with the CONSTRAINT line removed").
   `ExtractionConfig(null_twins=False)` gives independent null prompts instead.
6. **Exact duplicate prompts are redrawn.** Independent draws repeat a prompt in about one seed in seven.
7. **The verification grid is 100 items × 2 orders**, as in the budget estimate (`midterm_report/budget/`).

### How this bank differs from the earlier full bank (`jared/spar_horizon/full_bank.py`, `alan/ptm/prompts.py`)

The earlier seed-0 bank has the same 471/29 split but only phrasing 0, only `a)`/`b)` labels, no unit rewrites,
integer rounding above 100 (108 seconds and 139, 213, 326 years, where the report's Table 2 has 110, 140, 210, 330),
uniform rather than balanced horizons, independent null prompts, and no verification split. Its regime counts were
none 295 / both 93 / near 83 / null 29. The two banks are not interchangeable.

## The checker

A second model reads each prompt (and only the prompt) and reports, in JSON: the stated horizon; each option's
label, payout and timing; which options pay off within the horizon under the rule above; the correct answer; and
any problems, in fixed categories (`grammar`, `ambiguity`, `inconsistency`, `implausible`, `instructions`,
`other`). The exact instructions are `CHECKER_INSTRUCTIONS` in `pbank/checker.py`, and are written into every
checker's `.meta.json`. The same instructions will be used for every tier.

Its answer is compared with the generator's (`agrees`). **A disagreement means one of the two is wrong**, so both
cases need a look: the generator may have a bug, or the prompt may read ambiguously enough to mislead a careful
reader. Nothing is removed from the bank automatically.

`local:<Hugging Face id>` runs a local model with thinking on. `anthropic:<id>` and `openrouter:<id>` run a hosted
model at `--reasoning-effort medium`, with the JSON schema enforced by the provider; Claude models through the
Anthropic API use adaptive thinking and return a *summary* of their reasoning, which the review page shows. All
backends receive the same instructions. A Claude refusal is recorded as a failed request and retried on the next run.

## Files

```
data/<bank>/prompts.jsonl            the prompts (one JSON object per line)
data/<bank>/manifest.json            settings, counts, code commit, prompts SHA-256
data/<bank>/checks/<checker>.jsonl   one check per prompt (+ <checker>.meta.json: settings, model, versions)
data/<bank>/reviews/<reviewer>.jsonl human judgements from the review page, append-only
runs/<model>__<bank>/run.json        model, revision, software, settings, prompts SHA-256, times
runs/<model>__<bank>/responses.jsonl one response per prompt
runs/<model>__<bank>/summary.json    counts per split and regime
```

**prompts.jsonl** fields (`pbank/records.py`, `PromptRecord`): `uid` (`x0042`, `x0042-null`, `v017-far-first`);
`tier`; `split`; `text` (exactly what the model sees); `horizon` (as written), `horizon_grid_value` (its natural
form before any rewrite), `horizon_unit`, `horizon_years` (approximate, display only), `rewritten`,
`phrasing_id`; `near` and `far` (`reward`, `delay`, `delay_years`); `reward_multiplier`; `near_first`;
`labels` (display order); `regime`; `correct_option` (`near`/`far`/null); `correct_label`;
`horizon_equals_a_delay`; `h_over_far_delay`; `pair_id` (shared by a null twin and its source, and by the two
orders of a verification item); `notes`.

**checks/\*.jsonl** fields (`CheckResult`): `uid`, `checker`, `agrees`, `checker_answer`, `generator_answer`,
`verdict` (`ok`/`problem`), `problems`, `parsed` (the full JSON reply), `parse_error`, `request_error`,
`raw_answer`, `thinking`, `n_generated`, `hit_token_cap`. A rerun appends; the last row per `uid` is current.

**responses.jsonl** fields: `uid`, `model`, `thinking`, `answer` (text after the model closed its thinking; null if
it never did), `choice_label`, `parse_status` (`ok`, `format_failure`, `conflicting`, `no_answer`),
`n_choice_mentions`, `chosen_option`, `correct` (null when the prompt has no correct answer or no choice was
read), `n_generated`, `hit_token_cap`, `seconds`.

**reviews/\*.jsonl** fields: `time`, `reviewer`, `uid`, `target` (`prompt` or a run name), `verdict`
(`ok`/`problem`/`unsure`), `note`. The last line per (reviewer, uid, target) is that reviewer's current judgement.

## Reviewing

Start `review.py`, choose your name in the drop-down (names come from `reviewers.txt`; there is no login), and
use the filters to work through a slice: for example *Checker: disagree*, *Baseline: incorrect*, or *My review:
not reviewed by me*. On each prompt you can judge the prompt itself and each model's response separately. Your
judgements go to `data/<bank>/reviews/<your-name>.jsonl`; commit that file to share them. One file per reviewer
means reviews from several people merge without conflicts. The page listens on 127.0.0.1 only.

## How a model's answer is read

Thinking and answer are split at the model's first end-of-thinking token (`</think>` for Qwen3/Qwen3.5,
`<channel|>` for Gemma 4), found in the tokenizer; a model family with neither is refused. If the model never
closes its thinking, there is no answer and nothing is parsed from the deliberation. The choice is read from
"I choose: …" in the answer (`pbank/answers.py` lists the accepted forms); a missing or self-contradicting choice
is recorded as such, not guessed. **This parser is a heuristic: hand-audit a sample of its output for every new
model before relying on its counts.** The review page shows the raw answer next to the parsed choice for that.

## Code map

| file | what it does |
|---|---|
| `generate.py` | step 1: build the bank, run a checker |
| `baseline.py` | step 2: run a model on the bank |
| `review.py` | step 3: the review web page (templates in `templates/`) |
| `pbank/durations.py` | durations as written, exact comparison, the 48-point grid, unit rewrites |
| `pbank/records.py` | the prompt record, the decision rule, JSONL files, the prompts hash |
| `pbank/t0.py` | the T0 template and the two prompt sets |
| `pbank/checker.py` | the checker's instructions, schema, reading its reply, running it |
| `pbank/local_model.py` | a local Hugging Face model: batched greedy generation, thinking split |
| `pbank/hosted_model.py` | a hosted model through the Anthropic API or OpenRouter, same interface |
| `pbank/env.py` | API keys from a git-ignored `.env` file |
| `pbank/answers.py` | reading and scoring the model's choice |
| `pbank/annotations.py` | human review judgements |
| `pbank/provenance.py` | code commit and timestamps for output files |

## Tests

```
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest -q -s tests
```

No GPU or network needed. Covered: the grid against the report's Table 2 value by value; the report's sample prompt
byte for byte; bank composition, determinism and the regime of every prompt recomputed from its text fields;
rewrites (tolerance, cap, regime preservation); null twins; the verification grid (range, matched units, no ties,
paired orders, no overlap with the extraction split); the decision rule including ties; the answer parser; the
checker's reply handling and resume/retry logic (with a fake model); the Anthropic request and reply mapping,
refusals and API errors (with a fake client) and the `.env` loader; the thinking split with the real Gemma 4 and
Qwen3.5 tokenizers (skipped if they are not cached); and the review page end to end with Flask's test client.
`-s` shows each test's diagnostic output.
