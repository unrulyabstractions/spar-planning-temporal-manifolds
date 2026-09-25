#!/usr/bin/env python
"""Upload the shareable activation subsets (plus each run's index.parquet, meta.json, subset.json) and a
dataset card to a Hugging Face dataset repo.

Usage: upload_hf_subsets.py REPO_ID [--runs runs/a runs/b ...] [--public] [--commit HASH]
Creates the repo if needed (private unless --public). Idempotent: re-running uploads only changed files.
"""
import argparse, json, subprocess
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd

ap = argparse.ArgumentParser()
ap.add_argument("repo_id"); ap.add_argument("--runs", nargs="*", default=sorted(str(p.parent) for p in Path("runs").glob("*/subset.json")))
ap.add_argument("--public", action="store_true"); ap.add_argument("--commit", default=None)
a = ap.parse_args()
api = HfApi(); who = api.whoami()["name"]
api.create_repo(a.repo_id, repo_type="dataset", private=not a.public, exist_ok=True)
commit = a.commit or subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
ops, table = [], []
for r in a.runs:
    r = Path(r); n = r.name; meta = json.load(open(r / "subset.json"))
    files = [r / "index.parquet", r / "meta.json", r / "subset.json"] + sorted(r.glob("acts_subset_*.safetensors"))
    for f in files:
        ops.append(CommitOperationAdd(path_in_repo=f"runs/{n}/{f.name}", path_or_fileobj=str(f)))
    size = sum(f.stat().st_size for f in files) / 2**20
    table.append(f"| `{n}` | {meta['n']} | {meta['layers']} | {', '.join(meta['positions'])} | {size:.0f} MB |")
card = f"""---
license: mit
pretty_name: Planning Temporal Manifolds — Qwen3-14B activation subsets
tags: [interpretability, activations, time-horizon, qwen3]
---
# Planning Temporal Manifolds: Qwen3-14B residual-stream activation subsets

Companion data for Alan's SPAR Fall 2026 project (mentors: Ian Rios-Sialer, Justin Shenk, Shantanu
Darveshi), extending *Temporal Concepts and their Shape in Large Language Models* (arXiv:2606.05194).
Code and results: snapshot commit `{commit}` of `CodeReclaimers/SPAR-2026` (shared via the SPAR
repo `unrulyabstractions/spar-planning-temporal-manifolds`, folder `alan/`).

## Contents

| run | prompts | layers | positions | size |
|---|---|---|---|---|
{chr(10).join(table)}

Per run: `index.parquet` (one row per prompt: prompt text and parameters, generated text, parsed
choice, float32 label logits, `p_short`), `meta.json` (model, layer convention, position labels),
`subset.json` (layers, positions, row chunks), and `acts_subset_L{{layer}}_c{{chunk}}.safetensors`
holding tensor `acts` of shape `[rows, n_positions, 5120]` in bfloat16, rows in `index.parquet` order.

- Model: `Qwen/Qwen3-14B`, bfloat16, no-thinking chat template (empty think block pre-filled).
- Stored layer `l` = residual stream after decoder layer `l-1` (layer 0 = embeddings); no final norm.
- Positions: T0–T8 = the 9-token transition window `<|im_end|>`, `\\n`, `<|im_start|>`, `assistant`,
  `\\n`, `<think>`, `\\n\\n`, `</think>`, `\\n\\n`; R0–R3 = the first generated tokens (`I`, ` choose`,
  `:`, ` a`/` b`).
- Activations come from an unpadded per-sample forward pass and are bit-exact against
  `transformers` hidden states; a fresh capture reproduces them byte-for-byte.

## Loading

```python
from huggingface_hub import snapshot_download
from ptm.subset import load_subset          # from the code snapshot; or read the safetensors directly
d = snapshot_download("{a.repo_id}", repo_type="dataset")
X = load_subset(f"{{d}}/runs/qwen3-14b_investment_n2000_s0", layer=37, position="T3")   # float32 [2000, 5120]
```
Without the package: `safetensors.safe_open(file, "pt")["acts"]` and `subset.json` give the same rows.

Full activations (41 layers × 17 positions, ~14 GB per run) are not hosted; `scripts/reproduce_all.sh`
in the code snapshot regenerates every dataset, capture, and analysis (~1 h 45 on 2 × 16 GB GPUs).
"""
ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card.encode()))
print(f"uploading {len(ops)} files to {a.repo_id} as {who} ({'public' if a.public else 'private'}) …")
info = api.create_commit(a.repo_id, repo_type="dataset", operations=ops, commit_message=f"Activation subsets from SPAR-2026 @ {commit}")
print("done:", info.commit_url)
