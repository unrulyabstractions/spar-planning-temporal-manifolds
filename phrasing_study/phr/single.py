"""Run the single-turn protocols.

choice: prompt + "I choose:" (+ shared label prefix) in one forward pass; P(short) = two-way softmax of the short and
        long label logits. Residual stream stored at the transition window T0..T8 and at C (the last prefill token,
        which predicts the label). A stratified sample is also generated freely (no prefill) to check that the
        model writes "I choose: <label>" and that its label agrees with the readout (`gencheck.parquet`).
state:  the model is asked to name the horizon; greedy, short generation, parsed to years.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import save_file

from .durations import parse_years
from .model import (PREFILL, Capture, encode_chat, forward_choice, generate, label_readout, transition_window)

CHOICE_POSITIONS = [f"T{i}" for i in range(9)] + ["C"]
_CHOICE_RE = re.compile(r"I choose:\s*\**\s*\(?([A-Za-z0-9])\)?", re.IGNORECASE)


_SEP_RE = re.compile(r"^\s*I choose:(\s*[*_]*\s*)\(?[A-Za-z0-9]\)")


def detect_separator(tok, model, texts: list[str], batch: int = 16) -> tuple[str, dict]:
    """What the model writes between "I choose:" and the label when generating freely (e.g. " " or " **"), from a
    few canonical prompts. The readout prefills it, so the label token is the model's actual next token."""
    seps = []
    for b in range(0, len(texts), batch):
        for g in generate(tok, model, [encode_chat(tok, t) for t in texts[b:b + batch]], max_new=10):
            m = _SEP_RE.match(tok.decode(g, skip_special_tokens=True))
            seps.append(m.group(1) if m else None)
    counts = pd.Series([s if s is not None else "<no match>" for s in seps]).value_counts().to_dict()
    found = [s for s in seps if s is not None]
    return (max(set(found), key=found.count) if found else " "), counts


def _prep_choice(tok, items: pd.DataFrame, sep: str) -> list[dict]:
    cache, out = {}, []
    for r in items.itertuples():
        key = (r.label_short, r.label_long)
        if key not in cache:
            cache[key] = label_readout(tok, PREFILL + sep, r.label_short, r.label_long)
        pre, i_short, i_long = cache[key]
        ids = encode_chat(tok, r.text)
        win = transition_window(tok, ids)
        seq = ids + pre
        out.append(dict(item_id=r.item_id, seq=seq, pos=win + [len(seq) - 1], labels=(i_short, i_long)))
    return out


def run_choice(items: pd.DataFrame, out_dir: Path, tok, model, layers: list[int], batch: int = 16,
               shard_size: int = 2048, n_gencheck: int = 160, log=print) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    canon = items[items.variant == "canonical"]
    sep, counts = detect_separator(tok, model, canon.text.iloc[:: max(1, len(canon) // 48)].tolist()[:48], batch)
    write_meta(out_dir, choice_separator=sep, choice_separator_counts=counts)
    log(f"choice readout separator {sep!r} (free generations: {counts})")
    cap = Capture(model, layers)
    prepped = _prep_choice(tok, items, sep)
    order = sorted(range(len(prepped)), key=lambda i: len(prepped[i]["seq"]))     # similar lengths per batch
    res, acts, shard, t0 = [], [], 0, time.time()

    def flush():
        nonlocal acts, shard
        if acts:
            save_file({"acts": torch.cat(acts)}, str(out_dir / f"choice_acts_{shard:04d}.safetensors"))
            shard, acts = shard + 1, []

    n_in_shard = 0
    for b in range(0, len(order), batch):
        chunk = [prepped[i] for i in order[b:b + batch]]
        la, lb, mass, a = forward_choice(tok, model, cap, [c["seq"] for c in chunk], [c["pos"] for c in chunk],
                                         [c["labels"] for c in chunk])
        for j, c in enumerate(chunk):
            d = float(la[j] - lb[j])                       # logit(short) - logit(long)
            res.append(dict(item_id=c["item_id"], logit_diff=d, p_short=float(torch.sigmoid(torch.tensor(d))),
                            mass_ab=float(mass[j]), n_tokens=len(c["seq"]), shard=shard, row_in_shard=n_in_shard + j))
        acts.append(a)
        n_in_shard += len(chunk)
        if n_in_shard >= shard_size:
            flush()
            n_in_shard = 0
        if (b // batch) % 50 == 0:
            log(f"choice {b + len(chunk)}/{len(order)}  {time.time() - t0:.0f}s")
    flush()
    cap.remove()
    out = items.merge(pd.DataFrame(res), on="item_id")
    out.to_parquet(out_dir / "choice.parquet")
    log(f"choice done: {len(out)} items, {shard} shards, {time.time() - t0:.0f}s")

    # free-generation check on a stratified sample
    rng = np.random.default_rng(0)
    sample = (out.groupby("family", group_keys=False)
              .apply(lambda g: g.iloc[rng.permutation(len(g))[: max(4, n_gencheck // out.family.nunique())]]))
    gens = []
    for b in range(0, len(sample), batch):
        chunk = sample.iloc[b:b + batch]
        replies = generate(tok, model, [encode_chat(tok, t) for t in chunk.text], max_new=40)
        for r, g in zip(chunk.itertuples(), replies):
            text = tok.decode(g, skip_special_tokens=True)
            m = _CHOICE_RE.search(text)
            got = m.group(1) if m else None
            short, long = r.label_short.rstrip(")"), r.label_long.rstrip(")")
            gens.append(dict(item_id=r.item_id, gen_text=text, gen_label=got,
                             gen_short=None if got not in (short, long) else got == short,
                             readout_short=r.p_short > 0.5))
    g = pd.DataFrame(gens)
    g.to_parquet(out_dir / "gencheck.parquet")
    ok = g.gen_short.notna()
    log(f"gencheck: {ok.mean():.0%} follow the format; readout agrees on "
        f"{(g[ok].gen_short == g[ok].readout_short).mean():.0%} of those")


def run_state(items: pd.DataFrame, out_dir: Path, tok, model, batch: int = 16, log=print) -> None:
    rows = []
    for b in range(0, len(items), batch):
        chunk = items.iloc[b:b + batch]
        replies = generate(tok, model, [encode_chat(tok, t) for t in chunk.text], max_new=24)
        for r, g in zip(chunk.itertuples(), replies):
            text = tok.decode(g, skip_special_tokens=True).strip()
            rows.append(dict(item_id=r.item_id, answer=text, stated_years=parse_years(text)))
    out = items.merge(pd.DataFrame(rows), on="item_id")
    out.to_parquet(out_dir / "state.parquet")
    log(f"state done: {len(out)} items, parsed {out.stated_years.notna().mean():.0%}")


def write_meta(out_dir: Path, **kw) -> None:
    p = out_dir / "capture_meta.json"
    meta = json.loads(p.read_text()) if p.exists() else {}
    meta.update(kw)
    p.write_text(json.dumps(meta, indent=1, default=str))
