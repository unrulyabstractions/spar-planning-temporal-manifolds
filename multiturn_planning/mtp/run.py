"""Generate the conversations turn by turn and capture residual activations at the named positions.

Per turn k (1 outline, 2..6 steps, 7 "Plan Completed"):
  1. render the history with the chat template (thinking off), batch-generate reply k (left padding;
     greedy for the horizon condition, sampled with Qwen's non-thinking settings for the no-horizon condition);
     step replies are prefilled with "Step <n>:" (after the pre-reply window, so P0..P8 are unaffected);
  2. re-run prompt_k + reply_k teacher-forced, unbatched (no padding), and keep the residual stream at the
     named positions for the selected layers.

Residual convention (as in alan/ptm/capture.py): layer 0 = embedding output, layer l = output of decoder
layer l-1; no final norm. Activations are stored as float16 [n_turns, n_layers_kept, n_positions, d_model]
in safetensors shards with a bool mask for positions that don't exist in that turn (H/V outside step turns,
E on truncated replies). One row per (conversation, turn) in index.parquet.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from .chat import POSITIONS, clean_reply, cut_step_reply, encode_prompt, messages_for_turn, turn_positions
from .prompts import N_STEPS, SYSTEM, ConversationSpec, outline_mentions_time, parse_step, step_prefill

N_TURNS = N_STEPS + 2
MAX_NEW = {1: 200, **{k: 220 for k in range(2, N_STEPS + 2)}, N_STEPS + 2: 16}
SAMPLING = dict(do_sample=True, temperature=0.7, top_p=0.8, top_k=20)   # Qwen3 non-thinking recommendation


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).parent).stdout.strip() or None
    except OSError:
        return None


def turn_kind(k: int) -> str:
    return "outline" if k == 1 else ("done" if k == N_TURNS else "step")


def default_layers(n_layers: int) -> list[int]:
    """Every 4th residual layer plus the proposal's 0.4 / 0.6 / 0.8 L."""
    keep = set(range(0, n_layers + 1, 4)) | {round(f * n_layers) for f in (0.4, 0.6, 0.8)} | {n_layers}
    return sorted(keep)


class Capture:
    """Forward hooks on the embedding and every decoder layer; keeps chosen layers at chosen positions."""

    def __init__(self, model, layers: list[int]):
        mods = [model.model.embed_tokens] + list(model.model.layers)
        self.layers, self.positions, self.out = layers, None, {}
        self.handles = [mods[l].register_forward_hook(self._hook(l)) for l in layers]

    def _hook(self, l):
        def fn(_m, _inp, output):
            h = output[0] if isinstance(output, (tuple, list)) else output
            if self.positions is not None:
                self.out[l] = h[0, self.positions].to(torch.float16).cpu()
        return fn

    @torch.no_grad()
    def __call__(self, model, ids: list[int], positions: list[int]) -> torch.Tensor:
        self.positions, self.out = torch.tensor(positions), {}
        model(input_ids=torch.tensor([ids], device=model.device))
        self.positions = None
        return torch.stack([self.out[l] for l in self.layers])          # [n_layers_kept, n_pos, d]


def load(model_name: str, dtype=torch.bfloat16, gpu_gib: float | None = None):
    """gpu_gib: cap GPU memory and offload the rest to CPU (local tests of models that don't fit)."""
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    placement = (dict(device_map="auto", max_memory={0: f"{gpu_gib}GiB", "cpu": "24GiB"}) if gpu_gib
                 else dict(device_map="cuda"))
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, **placement)
    model.eval()
    return tok, model


@torch.no_grad()
def generate_batch(tok, model, prompts: list[list[int]], max_new: int, greedy: bool, seed: int) -> list[list[int]]:
    """Reply ids per prompt, cut after the first <|im_end|> (kept) or at max_new (truncated)."""
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    batch = tok.pad({"input_ids": prompts}, return_tensors="pt").to(model.device)
    kw = dict(do_sample=False) if greedy else SAMPLING
    if not greedy:
        torch.manual_seed(seed)
    out = model.generate(**batch, max_new_tokens=max_new, pad_token_id=tok.pad_token_id,
                         eos_token_id=[im_end, tok.eos_token_id], **kw)
    replies = []
    for row in out[:, batch["input_ids"].shape[1]:].tolist():
        replies.append(row[: row.index(im_end) + 1] if im_end in row else
                       [t for t in row if t != tok.pad_token_id])
    return replies


def run(specs: list[ConversationSpec], out_dir: Path, model_name: str, batch_size: int = 16,
        layers: list[int] | None = None, shard_size: int = 256, gpu_gib: float | None = None, log=print) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tok, model = load(model_name, gpu_gib=gpu_gib)
    n_layers = model.config.num_hidden_layers
    layers = layers or default_layers(n_layers)
    cap = Capture(model, layers)
    log(f"model {model_name}: {n_layers} layers, d={model.config.hidden_size}; keeping layers {layers}; "
        f"{len(specs)} conversations x {N_TURNS} turns")
    replies = {s.conv_id: [] for s in specs}
    rows, acts, masks, shard = [], [], [], 0
    t0 = time.time()

    def flush():
        nonlocal acts, masks, shard
        if acts:
            save_file({"acts": torch.stack(acts), "valid": torch.tensor(np.array(masks))},
                      str(out_dir / f"acts_{shard:04d}.safetensors"))
            shard, acts, masks = shard + 1, [], []

    for k in range(1, N_TURNS + 1):
        for greedy in (True, False):
            group = [s for s in specs if s.greedy == greedy]
            for b in range(0, len(group), batch_size):
                chunk = group[b:b + batch_size]
                prompts = [encode_prompt(tok, messages_for_turn(s.first_user, replies[s.conv_id])) for s in chunk]
                pre = tok.encode(step_prefill(k - 1), add_special_tokens=False) if turn_kind(k) == "step" else []
                gen = generate_batch(tok, model, [p + pre for p in prompts], MAX_NEW[k], greedy, seed=1000 * k + b)
                for s, p, r in zip(chunk, prompts, gen):
                    r = pre + r                                   # the reply as it stands in the conversation
                    r, cut = cut_step_reply(tok, r) if turn_kind(k) == "step" else (r, False)
                    pos = turn_positions(tok, p, r)
                    valid = [pos[n] is not None for n in POSITIONS]
                    idx = [pos[n] if pos[n] is not None else 0 for n in POSITIONS]
                    acts.append(cap(model, p + r, idx))
                    masks.append(valid)
                    text = clean_reply(tok.decode(r))
                    replies[s.conv_id].append(text)
                    step = parse_step(text) if turn_kind(k) == "step" else {}
                    rows.append({**s.to_dict(), "turn": k, "kind": turn_kind(k), "shard": shard,
                                 "row_in_shard": len(acts) - 1, "prompt_len": len(p), "n_reply": len(r),
                                 "truncated": pos["E"] is None, "cut": cut, "reply": text,
                                 "outline_mentions_time": outline_mentions_time(text) if k == 1 else None,
                                 "outline_full": ("time horizon" in text.lower()) if k == 1 else None,
                                 "step_no": step.get("step_no"), "h_step_text": step.get("h_step_text"),
                                 "h_step_years": step.get("h_step_years"),
                                 "positions": json.dumps(pos)})
                    if len(acts) == shard_size:
                        flush()
                log(f"turn {k} {'greedy' if greedy else 'sampled'} {b + len(chunk)}/{len(group)}  "
                    f"{time.time() - t0:.0f}s")
    flush()
    df = pd.DataFrame(rows).drop(columns=["first_user"])
    df.to_parquet(out_dir / "index.parquet")
    pd.DataFrame([s.to_dict() for s in specs]).to_parquet(out_dir / "conversations.parquet")
    (out_dir / "capture_meta.json").write_text(json.dumps(
        {"model": model_name, "model_revision": getattr(model.config, "_commit_hash", None),
         "git_commit": _git_commit(), "torch": torch.__version__, "transformers": transformers.__version__,
         "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
         "n_conversations": len(specs), "batch_size": batch_size, "n_layers": n_layers, "d_model": model.config.hidden_size, "layers": layers,
         "positions": POSITIONS, "n_turns": N_TURNS, "system": SYSTEM, "step_prefill": step_prefill(0).replace("0", "<n>"), "max_new": MAX_NEW, "sampling": SAMPLING,
         "wall_seconds": round(time.time() - t0, 1)}, indent=1))
    log(f"done: {len(df)} turn rows in {shard} shards, {time.time() - t0:.0f}s")
