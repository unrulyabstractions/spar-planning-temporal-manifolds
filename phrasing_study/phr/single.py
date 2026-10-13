"""Run the single-turn protocols.

choice: prompt + "I choose:" + separator, then the label. The model may write " " or " **" before the label, so each
        label's probability is summed over both separator paths (READOUT_SEPS), each path's probability counted from
        "I choose:" on: P(label) = sum_s P(s | "I choose:") P(label | "I choose:" s). P(short) = P_short / (P_short +
        P_long); mass_ab = P_short + P_long. Per-separator values are kept (ld_space, mass_space, ld_bold, mass_bold).
        Residual stream stored at the transition window T0..T8 and at C (the last prefill token of the separator the
        model writes most often, detected per run, which predicts the label). A stratified sample is also generated freely (no prefill) to check that the
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
from .model import (Capture, encode_chat, forward_logprobs, generate, plan_readout, readout_paths, transition_window)

CHOICE_POSITIONS = [f"T{i}" for i in range(9)] + ["C"]
_CHOICE_RE = re.compile(r"I choose:\s*\**\s*\(?([A-Za-z0-9])\)?", re.IGNORECASE)


READOUT_SEPS = (" ", " **")
SEP_NAMES = {" ": "space", " **": "bold"}
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


def _sep_name(sep: str, i: int) -> str:
    return SEP_NAMES.get(sep, f"sep{i}")


def _prep_choice(tok, items: pd.DataFrame, seps: list[str]) -> list[dict]:
    """One row per forward sequence (letter labels: 1 per item; ' 1)' vs ' **1)': 2). seps[0] is the main separator:
    its sequence carries the stored activations, C = its last prefill token."""
    cache, rows = {}, []
    for n, r in enumerate(items.itertuples()):
        key = (r.label_short, r.label_long)
        if key not in cache:
            paths = readout_paths(tok, seps, r.label_short, r.label_long)
            cache[key] = (paths, *plan_readout([p[1] for p in paths]))
        paths, pres, where, k = cache[key]
        ids = encode_chat(tok, r.text)
        win, L = transition_window(tok, ids), len(ids)
        for j, pre in enumerate(pres):
            reads, keys = [], []
            for i, (sep, ppre, ia, ib) in enumerate(paths):
                if where[i] != j:
                    continue
                reads += [(L + t - 1, ppre[t]) for t in range(k, len(ppre))]      # the separator's own tokens
                keys += [(sep, "sep")] * (len(ppre) - k)
                reads += [(L + len(ppre) - 1, ia), (L + len(ppre) - 1, ib)]
                keys += [(sep, "a"), (sep, "b")]
            main = where[0] == j
            c = L + len(paths[0][1]) - 1 if main else L + len(pre) - 1
            rows.append(dict(n=n, item_id=r.item_id, seq=ids + pre, pos=win + [c], reads=reads, keys=keys, main=main,
                             n_tokens=L + len(paths[0][1])))
    return rows


def _combine(lp: dict, names: dict) -> dict:
    """lp[sep] = {"sep": log P(separator tokens), "a": [...], "b": [...]} -> summed readout + per-separator columns."""
    out, la, lb = {}, [], []
    for sep, d in lp.items():
        a, b = d["sep"] + d["a"], d["sep"] + d["b"]
        la.append(a)
        lb.append(b)
        out[f"ld_{names[sep]}"], out[f"mass_{names[sep]}"] = a - b, float(np.exp(a) + np.exp(b))
    LA, LB = np.logaddexp.reduce(la), np.logaddexp.reduce(lb)
    out.update(logit_diff=float(LA - LB), p_short=float(1 / (1 + np.exp(LB - LA))), mass_ab=float(np.exp(LA) + np.exp(LB)))
    return out


def run_choice(items: pd.DataFrame, out_dir: Path, tok, model, layers: list[int], batch: int = 16,
               shard_size: int = 2048, n_gencheck: int = 160, log=print) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    canon = items[items.variant == "canonical"]
    sep, counts = detect_separator(tok, model, canon.text.iloc[:: max(1, len(canon) // 48)].tolist()[:48], batch)
    seps = [sep] + [s for s in READOUT_SEPS if s != sep]
    names = {s: _sep_name(s, i) for i, s in enumerate(seps)}
    write_meta(out_dir, choice_separator=sep, choice_separator_counts=counts, choice_readout_separators=seps,
               choice_readout="sum over separators")
    log(f"choice readout separator {sep!r} (free generations: {counts}); readout sums {seps}")
    cap = Capture(model, layers)
    rows = _prep_choice(tok, items, seps)
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["seq"]))     # similar lengths per batch
    lp = [dict() for _ in range(len(items))]
    where, acts, shard, n_in_shard, t0 = {}, [], 0, 0, time.time()

    def flush():
        nonlocal acts, shard
        if acts:
            save_file({"acts": torch.cat(acts)}, str(out_dir / f"choice_acts_{shard:04d}.safetensors"))
            shard, acts = shard + 1, []

    for b in range(0, len(order), batch):
        chunk = [rows[i] for i in order[b:b + batch]]
        vals, a = forward_logprobs(tok, model, cap, [c["seq"] for c in chunk], [c["pos"] for c in chunk],
                                   [c["reads"] for c in chunk])
        mains = [j for j, c in enumerate(chunk) if c["main"]]
        for j, c in enumerate(chunk):
            for (s, what), v in zip(c["keys"], vals[j].tolist()):
                d = lp[c["n"]].setdefault(s, {"sep": 0.0})
                if what == "sep":
                    d["sep"] += v
                else:
                    d[what] = v
        for m, j in enumerate(mains):
            where[chunk[j]["n"]] = dict(item_id=chunk[j]["item_id"], n_tokens=chunk[j]["n_tokens"], shard=shard,
                                        row_in_shard=n_in_shard + m)
        if mains:
            acts.append(a[mains])
            n_in_shard += len(mains)
        if n_in_shard >= shard_size:
            flush()
            n_in_shard = 0
        if (b // batch) % 50 == 0:
            log(f"choice {b + len(chunk)}/{len(order)}  {time.time() - t0:.0f}s")
    flush()
    cap.remove()
    res = [dict(**where[n], **_combine(lp[n], names)) for n in range(len(items))]
    out = items.merge(pd.DataFrame(res), on="item_id")
    out.to_parquet(out_dir / "choice.parquet")
    log(f"choice done: {len(out)} items ({len(rows)} sequences), {shard} shards, {time.time() - t0:.0f}s; "
        + ", ".join(f"mass_{v} median {out[f'mass_{v}'].median():.3f}" for v in names.values())
        + f", summed {out.mass_ab.median():.3f}")

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
