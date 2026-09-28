---
# License of this dataset card and the derived data. MIT matches Alan's companion dataset; change before uploading if
# the group prefers otherwise. The activations are derived from Qwen/Qwen3-14B (Apache-2.0).
license: mit
pretty_name: "PTM relevance-aware horizon decoder: Qwen3-14B residual-stream activations with distractor durations"
language:
- en
tags:
- interpretability
- mechanistic-interpretability
- activations
- residual-stream
- probing
- time-horizon
- qwen3
size_categories:
- 1K<n<10K
configs:
- config_name: index
  default: true
  data_files:
  - split: prompts
    path: runs/qwen3-14b_relevance_s0/index.parquet
- config_name: prompt_design
  data_files:
  - split: prompts
    path: data/prompts_s0.parquet
---

# Relevance-aware horizon decoder: Qwen3-14B activations with irrelevant durations

Residual-stream activations of **Qwen3-14B** on 5,520 short decision prompts, captured for the *relevance-aware
horizon decoder* experiment of the SPAR (fall 2026) project **Planning Temporal Manifolds** (PTM). Half the prompts
contain one extra sentence with a duration that is irrelevant to the decision. The data lets you ask whether a linear
readout of the model's **time horizon** can be made to ignore such durations.

This card documents the data only. Design, code and analysis live in the experiment folder
[`relevance_aware_decoder/`](https://github.com/unrulyabstractions/spar-planning-temporal-manifolds/tree/relevance-aware-decoder/relevance_aware_decoder)
of the project repository (branch `relevance-aware-decoder`).

## Background

PTM asks whether the time horizon an LLM is planning over can be read from its residual stream at turn-boundary tokens.
It extends *Temporal Concepts and their Shape in Large Language Models* (arXiv:2606.05194).

**Prior work (Alan).** On Qwen3-14B, Alan (a) replicated the preprint's horizon geometry; (b) trained ridge decoders
from single layer × token cells to log horizon on a matched prompt matrix; and (c) found that these decoders are
**pulled toward irrelevant durations** that the model's *choice* ignores. His "mention" control ("… was established
20 years ago") and "role" control ("You have 2 hours to prepare …") barely move the choice (logistic coefficient −0.03
and −0.11, versus −1.7 to −2.0 for the real horizon), yet correlate with the decoder's error (+0.58 / +0.74 at layer 26,
token T1). He listed a decoder trained *with* distractors as a next step. His activations and code:
[CodeReclaimers/spar-planning-temporal-manifolds-activations](https://huggingface.co/datasets/CodeReclaimers/spar-planning-temporal-manifolds-activations)
and the `alan/` folder of the project repository.

**This experiment** reuses Alan's `ptm` package (prompt matrix, chat encoding, capture hooks, storage) unchanged. It
adds five families of distractor sentences, a clean twin for every distractor prompt, and decoders trained with
distractors, tested on (A) trained sentences, (B) new wording and (C) a family held out of training.

## Contents

| Path | What |
|---|---|
| `runs/qwen3-14b_relevance_s0/index.parquet` | one row per prompt (5,520): prompt, generation, parsed choice, choice logits. Same row order as the activations. |
| `runs/qwen3-14b_relevance_s0/acts_subset_L{LL}_c{k}.safetensors` | activations: 7 layers × 6 row chunks = 42 files, tensor `acts`, bf16 |
| `runs/qwen3-14b_relevance_s0/subset.json` | layers, positions, row chunks, `d_model`, layer convention |
| `runs/qwen3-14b_relevance_s0/meta.json` | model, capture settings, all 17 captured position labels |
| `runs/qwen3-14b_relevance_s0/manifest.json` | provenance: git commit, prompt-file hash, model revision, `pip freeze`, torch/CUDA, GPU, pipeline settings, timestamps |
| `runs/qwen3-14b_relevance_s0/SHA256SUMS` | checksums of every file below except itself and this card |
| `runs/qwen3-14b_relevance_s0/SHA256SUMS.shards` | checksums of the **full** activation shards, which are *not* included (see below) |
| `runs/qwen3-14b_relevance_s0/{pipeline,gen_prompts}.log`, `runs/qwen3-14b_relevance_s0.{capture,evaluate}.log` | logs of the run |
| `data/prompts_s0.parquet` (+ `.json` summary) | the prompt design: every prompt with its distractor family, template, slot, clean twin and gap |
| `data/prompts_s0_smoke.parquet` | the 616-prompt slice used for a local plumbing test (no activations here) |
| `results/qwen3-14b_relevance_s0/` | evaluation outputs: per-cell sweep, headline cells with bootstrap intervals, behavior gate, text baselines, all decoder predictions (`predictions.npz`) |

About 4.0 GB in total, almost all of it activations. The layout mirrors the experiment folder, so a download can be
checked with `sha256sum -c runs/qwen3-14b_relevance_s0/SHA256SUMS` from its root, and the analysis script
(`scripts/evaluate.py … --acts subset`) can be run on it.

## Provenance

| | |
|---|---|
| Code | commit `843a4af` (branch `relevance-aware-decoder`), clean working tree |
| Model | `Qwen/Qwen3-14B`, revision `40c069824f4251a91eefaf281ebe4c544efd3e18`, bf16, thinking disabled (empty think block) |
| Decoding | greedy, up to 48 new tokens; batch 16, left-padded for generation |
| Activations | from a separate **unpadded** teacher-forced pass per prompt; a pre-run check matched `transformers` hidden states exactly (relative error 0) and the choice logits |
| Software | Python 3.12.14, torch 2.11.0+cu128, transformers 5.17.0 (full `pip freeze` in `manifest.json`) |
| Hardware | 1× NVIDIA RTX PRO 5000 Blackwell (48 GB), rented on vast.ai; capture of 5,520 prompts took 23.5 min |
| Date | 2026-09-28 |

## Prompts

Each prompt is a two-option intertemporal choice (small reward soon vs. large reward later) with an explicit time
horizon **H**, from Alan's matched matrix: 30 option configurations × 2 domains (household investment, city climate
policy) × 12 horizons (1 day … 100 years) × 3 renderings (structured, plain prose, varied prose) = **2,160 clean
prompts** (`condition == "main"`). Configurations are split 20 train / 4 dev / 6 test; 4 horizons (2 weeks, 3 months,
5 years, 20 years) are held out of training.

**3,360 distractor prompts** (`condition == "distractor"`) add one sentence with an irrelevant duration **D** to a
plain-prose prompt, at one of three places (`slot`). Each has a **clean twin** (`twin_uid`): the same prompt without
the sentence. D is never equal to H. Five families, one temporal role each, 672 prompts per family:

| `family` | example sentence (seen template) | D range |
|---|---|---|
| `entity_age` | The household's savings account was established {D} ago. | 1 day – 100 years |
| `prep_time` | You have {D} to prepare your recommendation. | 1 hour – 1 month |
| `tenure` | You have held this role for {D}. | 1 day – 30 years |
| `background_event` | The office building was last renovated {D} ago. | 1 day – 30 years |
| `other_horizon` | For an unrelated decision last year, the household used a time horizon of {D}. | 1 day – 100 years |

Each family has 2 `seen` templates (allowed in training) and 2 `unseen` templates (test only; `template_group`).
Distractor rows: 1,600 train, 320 dev (seen only), 1,440 test (720 seen + 720 unseen). All templates:
`data/prompts_s0.json`.

### Columns

`index.parquet` (from the capture) and `data/prompts_s0.parquet` (the design) have the same rows in the same order.

| Column(s) | Meaning |
|---|---|
| `sample_uid` | row key; `m0_…` clean, `r0_…` distractor |
| `condition`, `split`, `rendering` | `main`/`distractor`; `train`/`dev`/`test` (by option configuration); prompt rendering |
| `config_id`, `domain`, `scenario_id` | option configuration, domain, and their combination (the bootstrap cluster) |
| `horizon_text`, `horizon_years`, `horizon_heldout` | the stated horizon H |
| `distractor_text`, `distractor_years` | D (distractor rows only) |
| `short_reward`, `short_delay_*`, `long_reward`, `long_delay_*`, `short_first`, `label_a`, `label_b` | the two options and their order |
| `text` | the user message (wrapped in Qwen3's chat template at capture time) |
| `gen_text`, `gen_ids`, `n_generated` | the model's greedy answer |
| `choice`, `chose_short` | parsed choice (`a`/`b`) and whether it is the short option; the format was followed in 5,520/5,520 prompts |
| `logit_a`, `logit_b`, `p_short` | float32 logits of the two label tokens at the choice position; `p_short` = sigmoid of the short−long difference |
| `p_a`, `p_b`, `p_short_vocab`, `p_long_vocab` | full-vocabulary softmax mass on the label tokens |
| `transition_tokens`, `response_tokens`, `pos_valid` | the decoded tokens at the captured positions (JSON) |
| `shard`, `row` | location in the full shards (not included) |
| *design only:* `family`, `template_group`, `template_id`, `template`, `slot`, `twin_uid`, `log_gap` | distractor design; `log_gap` = log10(D / H) |

## Activations

- Tensor `acts` in each file: shape `[rows_in_chunk, 10, 5120]`, **bfloat16**. Chunks (`subset.json`) cover rows
  0–971, 972–1943, …, 4860–5519 of `index.parquet` (972 rows each, 660 in the last).
- **Layers** 14, 18, 22, 26, 29, 33, 37 of 40. Layer convention: layer 0 = embeddings; layer *l* = residual stream
  after decoder layer *l*−1 (resid_post); no final norm.
- **Positions** (second axis, in this order), identical for every prompt:

  | T0 | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | R0 |
  |---|---|---|---|---|---|---|---|---|---|
  | `<\|im_end\|>` | `\n` | `<\|im_start\|>` | `assistant` | `\n` | `<think>` | `\n\n` | `</think>` | `\n\n` | `I` |

  T0–T8 are the turn-boundary tokens after the user message (with the empty think block); R0 is the model's first
  generated token.

**Not included:** the full capture (all 41 layers × 17 positions: T0–T8 and R0–R7, about 39 GB). It was deleted with
the rented machine. `SHA256SUMS.shards` records its hashes, so a recapture with the same code, model revision and
software can be checked for byte-identity.

### Loading

Plain `numpy` + `pandas` (no torch needed; bf16 is widened to float32 exactly):

```python
import json
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

root = Path(snapshot_download("anicola/ptm-relevance-aware-decoder-qwen3-14b", repo_type="dataset"))   # ~4 GB; use allow_patterns to fetch less
run = root / "runs/qwen3-14b_relevance_s0"
sub = json.loads((run / "subset.json").read_text())


def load_bf16(path, name="acts"):
    """Read one bf16 tensor from a .safetensors file as float32, with numpy only."""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        meta = json.loads(f.read(n))[name]
        start, end = meta["data_offsets"]
        f.seek(8 + n + start)
        raw = np.frombuffer(f.read(end - start), dtype="<u2")
    return (raw.astype(np.uint32) << 16).view(np.float32).reshape(meta["shape"])


def acts(layer, position):
    """float32 [5520, 5120] for one layer x position, rows in index.parquet order."""
    p = sub["positions"].index(position)
    return np.concatenate([load_bf16(run / f"acts_subset_L{layer:02d}_c{k}.safetensors")[:, p]
                           for k in range(len(sub["chunks"]))])


index = pd.read_parquet(run / "index.parquet")
design = pd.read_parquet(root / "data/prompts_s0.parquet")
df = index.join(design[["family", "template_group", "slot", "twin_uid", "log_gap"]])   # same row order
X = acts(26, "T1")                              # (5520, 5120)
y = np.log10(df.horizon_years.to_numpy())       # decoding target
```

With torch installed, `safetensors.torch.load_file(path)["acts"].float()` gives the same values. With the project
repository on the path, `ptm.subset.load_subset(run, layer, position)` (from `alan/ptm/subset.py`) does the same.

## Results in brief

On this data (details in `results/qwen3-14b_relevance_s0/` and the experiment folder):
- The model's choice ignores all five families (behavior shift toward D between −0.02 and +0.06 of the way).
- A decoder trained on clean prompts is pulled 7–33% of the way toward D (depending on layer and token) and loses
  about half its accuracy on distractor prompts. This replicates Alan's finding.
- Training with distractors removes the pull toward D for trained sentences and new wording, and mostly for held-out
  families. But accuracy on distractor prompts still drops for new wording and new families. At layer 26, token T1:
  within-2× accuracy 0.80 (new wording) and 0.67 (held-out family), versus 0.94–0.95 on the clean twins.

## Caveats

- Single turn, explicit horizons, one model, one seed (the seed re-draws D, slot and template only).
- In 351 of 5,520 answers the model wrote the label unspaced (`I choose: **a)`), mostly on clean structured/varied
  prompts. `logit_a`/`logit_b` then refer to the unspaced label tokens. In 108 of 3,360 distractor/twin pairs the two
  prompts differ in this respect, which slightly mixes token variants in paired behavior comparisons.
- The distractor sentences are always in plain-prose prompts; the clean prompts cover all three renderings.

## Acknowledgements and citation

SPAR fall 2026, project *Planning Temporal Manifolds*. Mentors: Ian Rios-Sialer, Shantanu Darveshi, Justin Shenk.
Builds directly on Alan's experiments and code (see Background). Author of this experiment: Augusto Nicola.

If you use this data, please cite the preprint the project extends and link this dataset:

```bibtex
@misc{ptm_relevance_aware_decoder_2026,
  title        = {Relevance-aware horizon decoder: Qwen3-14B residual-stream activations with distractor durations},
  author       = {Augusto Nicola},
  year         = {2026},
  howpublished = {Hugging Face dataset, anicola/ptm-relevance-aware-decoder-qwen3-14b},
  note         = {SPAR fall 2026, Planning Temporal Manifolds. Code: github.com/unrulyabstractions/spar-planning-temporal-manifolds, commit 843a4af}
}
```
