"""E5: attribution patching (Jared's addition). Thinking off only.

Pairs: same rewards and delays, a short horizon where only the near option
delivers vs a long horizon where the far option delivers too. Only pairs whose
prompts tokenize to the same length are used. Scores per (layer, position,
component) in both directions, averaged over pairs; heatmaps and a top-10.

    python experiments/exp5_attribution.py --model Qwen/Qwen3-8B --thinking off
"""

import itertools
import random

import numpy as np
import torch

from common import parse_args
from spar_horizon.attribution import COMPONENTS, aggregate, attribute_pair
from spar_horizon.behavior import only_near_delivers
from spar_horizon.model_io import chat_text, generate_batch, label_ids, load_model
from spar_horizon.plots import heatmap
from spar_horizon.prompts import DELAY_PAIRS, DELAY_YEARS, REWARD_PAIRS, choice_prompt
from spar_horizon.runs import Manifest, resolve_out_dir

EXP = "exp5"
PLOT_TAIL = 24


def horizon_phrases():
    """(years, text) for integer month and year counts. Qwen tokenizes digit by
    digit, so pairing phrases with the same digit count gives equal lengths."""
    out = [(n / 12, f"{n} months") for n in range(2, 24)]
    out += [(n, f"{n} years") for n in range(1, 100)]
    return out


def candidate_pairs(seed=0, n_pairs=48):
    """Short horizon (only the near option delivers) vs long horizon (the far
    option delivers), same rewards/delays, same digit count in the phrase."""
    rng = random.Random(seed)
    pairs = []
    for rewards, delays in itertools.product(REWARD_PAIRS, DELAY_PAIRS):
        near = [(h, t) for h, t in horizon_phrases() if only_near_delivers({"horizon": h, "delays": delays})]
        far = [(h, t) for h, t in horizon_phrases() if h >= DELAY_YEARS[delays[1]]]
        for (hs, ts), (hl, tl) in itertools.product(near, far):
            if len(ts.split()[0]) == len(tl.split()[0]):
                pairs.append((rewards, delays, hs, ts, hl, tl))
    rng.shuffle(pairs)
    return pairs[:n_pairs]


def run(cfg, out=None):
    if cfg.thinking != "off":
        raise SystemExit("E5 runs with thinking off only (see 02_execution_plan.md)")
    out_dir = resolve_out_dir(EXP, cfg, out)
    manifest = Manifest(EXP, cfg, out_dir)
    tokenizer, model, device = load_model(cfg)
    ids = label_ids(tokenizer, cfg=cfg)
    scores, used, skipped = [], [], 0
    for rewards, delays, hs, ts, hl, tl in candidate_pairs(cfg.seed, cfg.limit or 48):
        clean = tokenizer(chat_text(tokenizer, choice_prompt(rewards, delays, hs, horizon_str=ts), cfg),
                          return_tensors="pt")["input_ids"]
        corrupt = tokenizer(chat_text(tokenizer, choice_prompt(rewards, delays, hl, horizon_str=tl), cfg),
                            return_tensors="pt")["input_ids"]
        if clean.shape != corrupt.shape:
            skipped += 1
            continue
        # The model may put formatting (e.g. ' **') before the label. Take the
        # clean run's lead tokens and append them to both, so the metric is
        # read at the position that predicts the label.
        g = generate_batch(tokenizer, model, device, [choice_prompt(rewards, delays, hs, horizon_str=ts)], cfg)[0]
        lead = g.ids[g.prompt_len:g.answer_start].unsqueeze(0).cpu()
        clean = torch.cat([clean, lead], dim=1)
        corrupt = torch.cat([corrupt, lead], dim=1)
        s = attribute_pair(model, clean.to(device), corrupt.to(device), ids)
        scores.append(s)
        used.append({"rewards": rewards, "delays": delays, "short": ts, "long": tl,
                     "clean_metric": s.clean_metric, "corrupt_metric": s.corrupt_metric})
        if len(scores) % 10 == 0:
            print(f"  {len(scores)} pairs")
    if not scores:
        raise RuntimeError("no equal-length pairs found")
    agg, width = aggregate(scores)
    print(f"{len(scores)} pairs used, {skipped} skipped for length mismatch, width {width}")
    flips = sum(u["clean_metric"] > 0 > u["corrupt_metric"] for u in used)
    print(f"mean metric clean {np.mean([u['clean_metric'] for u in used]):.3f}  "
          f"corrupt {np.mean([u['corrupt_metric'] for u in used]):.3f}  "
          f"behavioral flips (a -> b): {flips}/{len(used)}")

    tail = tokenizer.convert_ids_to_tokens(clean[0, -width:].tolist())
    top = []
    for direction in ("noising", "denoising"):
        for c in COMPONENTS:
            M = agg[direction][c]
            np.save(out_dir / f"{direction}_{c}.npy", M)
            v = np.abs(M).max() or 1
            show = min(PLOT_TAIL, M.shape[1])
            heatmap(M[:, -show:], [str(l) for l in range(M.shape[0])], tail[-show:],
                    out_dir / f"{direction}_{c}.png",
                    f"{direction} {c} (rows = layers, last {show} positions)",
                    vmin=-v, vmax=v, fmt="{:.1e}", annotate=False)
            for l, p in zip(*np.unravel_index(np.argsort(-np.abs(M), axis=None)[:10], M.shape)):
                top.append({"direction": direction, "component": c, "layer": int(l),
                            "pos": int(p), "token": tail[p], "score": float(M[l, p])})
    top.sort(key=lambda t: -abs(t["score"]))
    print("\ntop components:")
    for t in top[:10]:
        print(f"  {t['direction']:9} {t['component']:10} L{t['layer']:<3} pos {t['pos']:<3} "
              f"{t['token']!r:>14}  {t['score']:+.3e}")
    manifest.add(n_pairs=len(scores), skipped=skipped, flips=flips, width=width, tail_tokens=tail,
                 pairs=used, top=top[:30])
    manifest.write()
    return out_dir


if __name__ == "__main__":
    args, cfg = parse_args(__doc__)
    run(cfg, args.out)
