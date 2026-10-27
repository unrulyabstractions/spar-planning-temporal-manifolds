"""E8: causal tests of the horizon code on the full bank (spec: 07_spec_steering.md).

    python experiments/exp8_steering.py --inputs steering/inputs_Qwen3-8B_answer_slot.npz \
        --tests ablate,steer,post,answer --out /workspace/results/horizon
    python experiments/exp8_steering.py --inputs steering/inputs_Qwen3.5-0.8B_smoke.npz --limit 4   # local smoke

Inputs come from analysis/steering_prepare.py. Tests:
  patch   T1  interchange patching map, thinking off
  ablate  T2  mean-ablate the top-k horizon-average subspace at the off cells
  steer   T3  line / curve / mean-difference / random steering at the off cells
  post    T4  T2 and T3 at the thinking-on cells, on the cached reasoning
  answer  T5  answer-slot patching across prompts, per layer, both modes
Cells whose inputs carry a `decision` direction (steering_prepare.py, 07 section 10)
are steered and ablated along the horizon/decision split instead of the probe line.
Thinking off is teacher-forced through "I choose:" and each prompt is averaged
with its option-swapped twin. Writes <out>/exp8_<inputs name>/results.json
(e.g. exp8_Qwen3-8B_answer_slot) and prints the success criteria of the spec,
fixed before the run.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from spar_horizon import steering as S  # noqa: E402
from spar_horizon.banks import build_records  # noqa: E402
from spar_horizon.config import Settings  # noqa: E402
from spar_horizon.full_bank import render_variant  # noqa: E402
from spar_horizon.model_io import label_ids, load_model, suffix_length  # noqa: E402
from spar_horizon.runs import code_hash, out_base, stamp  # noqa: E402

TESTS = ("patch", "ablate", "steer", "post", "answer")
ALPHAS = (-2, -1, 1, 2)
ABLATE_KS = (1, 2, 4, 8)
CUT_K = {"off": 2, "on": 4}          # the k at which "necessary" is judged (spec section 5)
DIRECTIONS = ("line", "curve", "mean difference", "random 0", "random 1")
# for cells with a decision direction (07 section 10): the line is inert at every cell
# so far and is dropped; the curve is split into its decision part and the rest
DECISION_DIRECTIONS = ("curve", "curve without decision", "decision part of curve", "decision",
                       "mean difference", "random 0")
N_RANDOM_SUBSPACES = 3
MAX_PAIRS = 96
MAX_ANSWER_PAIRS = {"off": 96, "on": 64}   # thinking-on pairs cost ~2.5k-token passes per layer
MIN_PAIR_GAP = 1.0                    # donor-recipient log-odds gap below which a pair carries no transfer signal


class Runner:
    """The model, the label tokens, and batched scoring of prompt sets."""

    def __init__(self, model, device, pad_id, batch_size, long_batch_size):
        self.model, self.device, self.pad_id = model, device, pad_id
        self.batch_size, self.long_batch_size = batch_size, long_batch_size

    def score(self, prompts, cell_label=None, layer=None, make_edit=None):
        """Near log-odds per prompt. A thinking-off set averages each prompt with its
        swapped twin; the same edit is applied to both."""
        if make_edit is None:
            # the unedited baseline goes through the same prefix-cache path as last-position
            # edits, so their differences carry no cached-vs-full numerical offset
            return self.score_many(prompts, "pre-label", None, [None])[0]
        out = []
        for variant in prompts["variants"]:
            positions = variant["positions"][cell_label] if make_edit is not None else None
            out.append(S.score(self.model, self.device, variant["ids"], variant["near_first"], variant["labels"],
                               self.pad_id, prompts["batch_size"], layer, positions, make_edit))
        return np.mean(out, axis=0)

    def score_many(self, prompts, cell_label, layer, make_edits):
        """Near log-odds [len(make_edits), n], one row per edit at `layer` (None: no
        edit). When the cell is every row's last position the edits share one prefix
        pass per batch (S.score_last_edits); otherwise each edit is a full pass."""
        out = []
        for variant in prompts["variants"]:
            positions = variant["positions"][cell_label]
            if all(p == [len(seq) - 1] for p, seq in zip(positions, variant["ids"])):
                out.append(S.score_last_edits(self.model, self.device, variant["ids"], variant["near_first"],
                                              variant["labels"], self.pad_id, prompts["batch_size"],
                                              [None if make is None else (layer, make) for make in make_edits]))
            else:
                out.append(np.stack([S.score(self.model, self.device, variant["ids"], variant["near_first"],
                                             variant["labels"], self.pad_id, prompts["batch_size"], layer,
                                             positions, make) for make in make_edits]))
        return np.mean(out, axis=0)


def load_inputs(path):
    """meta (json), cells {key: arrays}, and the cached reasoning arrays (or None)."""
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    z = np.load(path)
    cells = {}
    for c in meta["cells"]:
        key = f"{c['mode']}|{c['label']}|{c['layer']}"
        names = ("mean", "w", "basis", "curve_logh", "curve_coords", "caa") + (
            ("decision",) if f"{key}|decision" in z.files else ())
        cells[key] = {**c, **{name: z[f"{key}|{name}"] for name in names}, "b": float(z[f"{key}|b"])}
    names = ("ids", "offsets", "close", "think_len", "forced", "labels", "record")
    post = {name: z[f"post|{name}"] for name in names} if meta["n_post"] else None
    return meta, cells, post


def off_prompts(records, tokenizer, cfg, batch_size):
    """Thinking-off set: each prompt and its swapped twin, teacher-forced through the
    prefill, with the cell positions (suffix newline, pre-label) per row."""
    labels = [label_ids(tokenizer, cfg=cfg)] * len(records)
    n_suffix = suffix_length(tokenizer, records[0]["prompt"], cfg)
    variants = []
    for swap in (False, True):
        ids = [S.encode(tokenizer, render_variant(r, swap=True) if swap else r["prompt"], cfg) for r in records]
        newline = [[len(seq) - n_suffix + 1] for seq in ids]
        if any(tokenizer.decode([seq[p[0]]]) != "\n" for seq, p in zip(ids, newline)):
            raise ValueError("the token after <|im_end|> is not a newline; the suffix layout changed")
        variants.append({"ids": ids, "labels": labels,
                         "near_first": [(r["near_label"] == "a") != swap for r in records],
                         "positions": {"suffix newline": newline, "pre-label": [[len(seq) - 1] for seq in ids]}})
    return {"variants": variants, "batch_size": batch_size}


def on_prompts(post, records, batch_size):
    """Thinking-on set from the cached reasoning: ids up to the label, positions of
    </think> and of the pre-label token."""
    ids = [post["ids"][a:b].tolist() for a, b in zip(post["offsets"][:-1], post["offsets"][1:])]
    rows = [records[j] for j in post["record"]]
    variant = {"ids": ids, "labels": [tuple(int(t) for t in pair) for pair in post["labels"]],
               "near_first": [r["near_label"] == "a" for r in rows],
               "positions": {"think close": [[int(c)] for c in post["close"]],
                             "pre-label": [[len(seq) - 1] for seq in ids]}}
    return {"variants": [variant], "batch_size": batch_size}, rows


def directions(cell):
    return DECISION_DIRECTIONS if "decision" in cell else DIRECTIONS


def delta_rows(cell, direction, alpha, log_h):
    """Per-row displacement [n, d] for one direction and alpha (decades along the
    curve of horizon averages). Every direction moves each row as far as the
    curve does, so they differ in direction only, except the two parts of the
    curve, which add up to it."""
    if direction == "curve":
        return S.curve_delta(cell, log_h, alpha)
    if direction == "curve without decision":
        return S.without_direction(S.curve_delta(cell, log_h, alpha), cell["decision"])
    if direction == "decision part of curve":
        curve = S.curve_delta(cell, log_h, alpha)
        return curve - S.without_direction(curve, cell["decision"])
    if direction == "decision":
        # a longer horizon (alpha > 0) should lower the near log-odds, so step against the decision vector
        vector = -cell["decision"]
    elif direction == "line":
        vector = cell["w"]
    elif direction == "mean difference":
        vector = cell["caa"]
    else:   # "random <seed>"
        vector = np.random.default_rng(int(direction.split()[1])).normal(size=len(cell["w"]))
    return S.matched_deltas(cell, vector, alpha, log_h)


def adder(deltas, device):
    return lambda rows: S.Add(torch.tensor(deltas[rows], device=device, dtype=torch.float32))


def run_ablate(runner, prompts, cell, regimes, base):
    """T2/T4 ablation at one cell: removed share of the contrast for the horizon-average
    subspace at every k, and for random subspaces at the k where "necessary" is
    judged (the control the criterion needs; at other k it would only cost time).
    Returns (removed shares, edited log-odds per prompt)."""
    d = len(cell["mean"])
    mean = torch.tensor(cell["mean"], device=runner.device, dtype=torch.float32)
    available = [k for k in ABLATE_KS if k <= cell["basis"].shape[0]]
    cut = CUT_K[cell["mode"]] if CUT_K[cell["mode"]] in available else max(available)
    subspaces = []   # (k, name, basis), all scored in one call so they share each batch's prefix pass
    for k in available:
        subspaces.append((k, "horizon", cell["basis"][:k]))
        if k == cut:
            subspaces += [(k, f"random {s}", S.random_orthonormal(d, k, seed=s)) for s in range(N_RANDOM_SUBSPACES)]
            if "decision" in cell:
                subspaces += [(k, "horizon without decision", S.basis_without(cell["basis"][:k], cell["decision"])),
                              (k, "decision", cell["decision"][None, :])]
    edits = [S.MeanAblate(mean, torch.tensor(basis, device=runner.device, dtype=torch.float32))
             for _, _, basis in subspaces]
    edited_all = runner.score_many(prompts, cell["label"], cell["layer"], [lambda rows, e=e: e for e in edits])
    out, per_prompt = {}, {}   # removed share per k and subspace; edited log-odds per prompt, "k|subspace"
    for (k, name, _), edited in zip(subspaces, edited_all):
        out.setdefault(k, {})[name] = S.removed_fraction(base, edited, regimes)
        per_prompt[f"{k}|{name}"] = edited.tolist()
    return out, per_prompt


def run_steer(runner, prompts, cell, regimes, log_h, base, contrast):
    """T3/T4 steering at one cell: effect (in contrast units) per direction and alpha,
    for near- and both-regime prompts, the probe readout shift, and the per-prompt
    effect signed toward the rational answer (for the curve-vs-line criterion)."""
    out = {}
    regimes = np.asarray(regimes)
    conditions = [(direction, alpha) for direction in directions(cell) for alpha in ALPHAS]
    deltas = {c: delta_rows(cell, *c, log_h) for c in conditions}
    edited_all = runner.score_many(prompts, cell["label"], cell["layer"],
                                   [adder(deltas[c], runner.device) for c in conditions])
    for (direction, alpha), edited in zip(conditions, edited_all):
        effect = (edited - base) / contrast
        out.setdefault(direction, {})[alpha] = {"near": S.bootstrap_mean(effect[regimes == "near"]),
                                                "both": S.bootstrap_mean(effect[regimes == "both"]),
                                                "readout": float((deltas[direction, alpha] @ cell["w"]).mean()),
                                                "toward": S.toward_rational(effect, regimes, alpha).tolist(),
                                                "effect": effect.tolist()}
    return out


def run_answer(runner, variant, regimes, near_labels, limit):
    """T5: answer-slot patching across prompts. The recipient's last position (the
    token before the label) is overwritten at one layer with a donor's state there;
    the donor is in the other regime with the near option under the same label.
    Transfer fraction per layer, pairs whose log-odds differ by more than
    MIN_PAIR_GAP. The last position is the only one that lines up across
    different reasoning traces, so this is the patch that works with thinking on."""
    pairs = S.answer_pairs(regimes, near_labels, random.Random(0), limit)
    n_layers = len(runner.model.model.layers)
    transfer = np.full((n_layers, len(pairs)), np.nan)
    gaps = np.zeros(len(pairs))
    ids, labels, near_first = variant["ids"], variant["labels"], variant["near_first"]
    batch_size = runner.long_batch_size if max(len(s) for s in ids) > 1000 else runner.batch_size
    for start in tqdm(range(0, len(pairs), batch_size), desc="answer-slot batches", unit="batch"):
        batch = pairs[start:start + batch_size]
        rows = slice(start, start + len(batch))
        recipient = [ids[r] for r, _ in batch]
        same_near = [near_first[r] for r, _ in batch]
        donor_logits, captured = S.forward_capture_last(runner.model, runner.device, [ids[d] for _, d in batch],
                                                        runner.pad_id)
        lo_d = S.near_log_odds(donor_logits, same_near, [labels[d] for _, d in batch])
        # row 0: the unpatched recipient; row L: its last position replaced with the donor's at layer L
        edits = [None] + [(layer, lambda rows, v=captured[layer - 1]: S.Replace(v[rows]))
                          for layer in range(1, n_layers)]
        scored = S.score_last_edits(runner.model, runner.device, recipient, same_near,
                                    [labels[r] for r, _ in batch], runner.pad_id, len(batch), edits)
        lo_r = scored[0]
        gaps[rows] = lo_d - lo_r
        transfer[1:, rows] = (scored[1:] - lo_r) / (lo_d - lo_r)
    used = np.abs(gaps) > MIN_PAIR_GAP
    return {"n_pairs": len(pairs), "n_used": int(used.sum()), "mean_gap": float(np.abs(gaps).mean()),
            "transfer": [S.bootstrap_mean(transfer[layer, used]) for layer in range(n_layers)]}


def run_patch(runner, records, tokenizer, cfg, limit):
    """T1: transfer fraction per (layer, position group), thinking off, no swap twin
    (an additive order effect cancels in the donor-recipient difference)."""
    pairs = S.crossing_pairs(records, tokenizer, cfg, random.Random(0), limit)
    labels = label_ids(tokenizer, cfg=cfg)
    n_layers = len(runner.model.model.layers)
    names = [g[0] for g in S.patch_groups(pairs[0], tokenizer)]
    transfer = np.full((n_layers, len(names), len(pairs)), np.nan)
    gaps = np.zeros(len(pairs))
    for start in tqdm(range(0, len(pairs), runner.batch_size), desc="patch batches", unit="batch"):
        batch = pairs[start:start + runner.batch_size]
        near_first = [records[p["record"]]["near_label"] == "a" for p in batch]
        recipient, donor = [p["recipient"] for p in batch], [p["donor"] for p in batch]
        donor_logits, donor_hidden, offsets = S.forward(runner.model, runner.device, donor, runner.pad_id, hidden=True)
        recipient_logits, _, _ = S.forward(runner.model, runner.device, recipient, runner.pad_id)
        lo_d = S.near_log_odds(donor_logits, near_first, [labels] * len(batch))
        lo_r = S.near_log_odds(recipient_logits, near_first, [labels] * len(batch))
        gaps[start:start + len(batch)] = lo_d - lo_r
        groups = [S.patch_groups(p, tokenizer) for p in batch]
        for layer in range(1, n_layers):
            for g in range(len(names)):
                positions = [gr[g][1] for gr in groups]
                values = S.gather(donor_hidden[layer], positions, offsets)
                patched, _, _ = S.forward(runner.model, runner.device, recipient, runner.pad_id, layer=layer,
                                          positions=positions, edit=S.Replace(values))
                lo_p = S.near_log_odds(patched, near_first, [labels] * len(batch))
                transfer[layer, g, start:start + len(batch)] = (lo_p - lo_r) / (lo_d - lo_r)
    used = np.abs(gaps) > MIN_PAIR_GAP
    summary = [[S.bootstrap_mean(transfer[layer, g, used]) for g in range(len(names))] for layer in range(n_layers)]
    return {"groups": names, "n_pairs": len(pairs), "n_used": int(used.sum()), "mean_gap": float(np.abs(gaps).mean()),
            "transfer": summary}


def verdicts(mode, cell_results):
    """The spec's section 5 criteria for one cell's ablation and steering results."""
    out = {}
    ablate, steer = cell_results.get("ablate"), cell_results.get("steer")
    if ablate:
        k = CUT_K[mode] if CUT_K[mode] in ablate else max(ablate)
        horizon = ablate[k]["horizon"][0]
        worst_random = max(v[0] for name, v in ablate[k].items() if name.startswith("random"))
        out["necessary"] = {"k": k, "removed": horizon, "random_max": worst_random,
                            "pass": bool(horizon >= 0.5 and worst_random <= 0.1)}
    if steer and "curve without decision" in steer:
        out.update(decision_verdicts(ablate, steer))
    if steer and "line" in steer:
        near_up, both_down = steer["line"][1]["near"][0], steer["line"][-1]["both"][0]
        random_abs = np.mean([abs(steer[r][a][g][0]) for r in DIRECTIONS if r.startswith("random")
                              for a in (-2, 2) for g in ("near", "both")])
        out["sufficient_line"] = {"near_at_+1": near_up, "both_at_-1": both_down, "random_abs_at_2": float(random_abs),
                                  "pass": bool(near_up <= -0.25 and both_down >= 0.25 and random_abs <= 0.1)}
        diffs = np.concatenate([np.asarray(steer["curve"][a]["toward"]) - np.asarray(steer["line"][a]["toward"])
                                for a in ALPHAS])
        mean, low, high = S.bootstrap_mean(diffs)
        out["curve_beats_line"] = {"mean": mean, "low": low, "high": high, "pass": bool(mean >= 0.1 and low > 0)}
    return out


def toward_at_2(steer, direction):
    """Per-prompt effect toward the rational answer at |alpha| = 2, both signs pooled."""
    return np.concatenate([np.asarray(steer[direction][a]["toward"]) for a in (-2, 2)])


def decision_verdicts(ablate, steer):
    """07 section 10, fixed before the run: is the horizon content beyond the
    decision direction used? Steering effects toward the rational answer at
    |alpha| = 2, all 176 prompts."""
    curve = S.bootstrap_mean(toward_at_2(steer, "curve"))
    rest = S.bootstrap_mean(toward_at_2(steer, "curve without decision"))
    part = S.bootstrap_mean(toward_at_2(steer, "decision part of curve"))
    random = S.bootstrap_mean(toward_at_2(steer, "random 0"))
    out = {"horizon_beyond_decision": {
        "curve": curve[0], "without_decision": rest[0], "low": rest[1], "high": rest[2],
        "decision_part": part[0], "random": random[0],
        "pass": bool(rest[0] >= 0.05 and rest[1] > 0 and rest[0] >= curve[0] / 3)}}
    k = next((k for k, row in ablate.items() if "horizon without decision" in row), None) if ablate else None
    if k is not None:
        removed = ablate[k]["horizon without decision"]
        worst_random = max(v[0] for name, v in ablate[k].items() if name.startswith("random"))
        out["ablate_beyond_decision"] = {"k": k, "removed": removed[0], "low": removed[1],
                                         "decision_alone": ablate[k]["decision"][0], "random_max": worst_random,
                                         "pass": bool(removed[0] >= 0.1 and removed[1] > 0 and worst_random <= 0.05)}
    return out


def print_cell(key, result, contrast):
    print(f"\n=== {key}   (contrast {contrast:+.2f} log-odds)")
    for k, row in result.get("ablate", {}).items():
        rand = [v[0] for name, v in row.items() if name.startswith("random")]
        print(f"  ablate k={k}: horizon subspace removes {row['horizon'][0]:+.2f} "
              f"[{row['horizon'][1]:+.2f}, {row['horizon'][2]:+.2f}] of the contrast; random {np.round(rand, 2)}")
        for name in ("horizon without decision", "decision"):
            if name in row:
                print(f"           {name} removes {row[name][0]:+.2f} [{row[name][1]:+.2f}, {row[name][2]:+.2f}]")
    for direction, by_alpha in result.get("steer", {}).items():
        cells = "  ".join(f"a={a:+d}: near {v['near'][0]:+.2f} both {v['both'][0]:+.2f} (readout {v['readout']:+.1f})"
                          for a, v in by_alpha.items())
        print(f"  steer {direction:23} {cells}")
    for name, v in result.get("verdicts", {}).items():
        print(f"  {name:18} {'PASS' if v['pass'] else 'fail'}  {{{', '.join(f'{k}: {x:.2f}' if isinstance(x, float) else f'{k}: {x}' for k, x in v.items() if k != 'pass')}}}")


def main(args):
    meta, cells, post = load_inputs(args.inputs)
    cfg = Settings(model=meta["model"], thinking="off", prefill="I choose:", bank="full", n_prompts=meta["n_prompts"],
                   seed=meta["seed"], limit=meta["limit"], batch_size=args.batch_size)
    records_all = build_records(cfg)
    records = [records_all[i] for i in meta["selected"]]
    if [r["sample_uid"] for r in records] != meta["sample_uids"]:
        raise ValueError("the prompt bank differs from the one the inputs were prepared on")
    if args.limit:
        records = records[:args.limit]
        if post is not None:
            keep = post["record"] < args.limit
            spans = list(zip(post["offsets"][:-1], post["offsets"][1:]))
            kept = [spans[i] for i in np.flatnonzero(keep)]
            ids = np.concatenate([post["ids"][a:b] for a, b in kept]) if kept else np.array([], np.int32)
            post = {**{k: v[keep] for k, v in post.items() if k not in ("ids", "offsets")}, "ids": ids,
                    "offsets": np.r_[0, np.cumsum([b - a for a, b in kept])]}
            post = post if len(post["record"]) else None
    tokenizer, model, device = load_model(cfg)
    runner = Runner(model, device, tokenizer.pad_token_id, args.batch_size, args.long_batch_size)
    out_dir = out_base(args.out) / Path(args.inputs).stem.replace("inputs_", "exp8_", 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    tests = args.tests.split(",")
    results = {"inputs": str(args.inputs), "code_hash": code_hash(), "tests": tests, "n_records": len(records),
               "cells": {}, "contrast": {}, "phases": {}}

    def timed(name, fn):
        t0 = time.time()
        stamp(f"{name} ...")
        value = fn()
        results["phases"][name] = round(time.time() - t0, 1)
        stamp(f"{name} done, {results['phases'][name]} s")
        return value

    if "patch" in tests:
        results["patch"] = timed("T1 patch", lambda: run_patch(runner, records, tokenizer, cfg, args.limit or MAX_PAIRS))
        p = results["patch"]
        print(f"\nT1: {p['n_used']} of {p['n_pairs']} pairs differ by more than {MIN_PAIR_GAP} log-odds "
              f"(mean gap {p['mean_gap']:.2f}). Transfer fraction, rows = layers:")
        print("       " + " ".join(f"{name[:9]:>9}" for name in p["groups"]))
        for layer, row in enumerate(p["transfer"]):
            if layer:
                print(f"  L{layer:<3} " + " ".join(f"{v[0]:9.2f}" for v in row))

    # which cell tests run per mode: thinking off by name, thinking on all under "post"
    cell_tests = {"off": [t for t in ("ablate", "steer") if t in tests],
                  "on": ["ablate", "steer"] if "post" in tests else []}
    sets = []
    if cell_tests["off"] or "answer" in tests:
        sets.append(("off", off_prompts(records, tokenizer, cfg, args.batch_size), records))
    if cell_tests["on"] or "answer" in tests:
        if post is None:
            print("\nthinking-on tests skipped: the inputs carry no cached thinking-on reasoning")
        else:
            prompts, rows = on_prompts(post, records, args.long_batch_size)
            sets.append(("on", prompts, rows))
    results["answer"] = {}
    for mode, prompts, rows in sets:
        regimes = [r["regime"] for r in rows]
        log_h = np.log10([r["horizon"] for r in rows])
        if "answer" in tests:
            near_labels = [r["near_label"] for r in rows]
            limit = min(MAX_ANSWER_PAIRS[mode], args.limit or MAX_ANSWER_PAIRS[mode])
            a = results["answer"][mode] = timed(f"T5 answer-slot patch, thinking {mode}", lambda: run_answer(
                runner, prompts["variants"][0], regimes, near_labels, limit))
            print(f"\nT5 thinking {mode}: {a['n_used']} of {a['n_pairs']} pairs usable (mean gap {a['mean_gap']:.2f}). "
                  "Transfer at the answer slot by layer:\n  " +
                  "  ".join(f"L{layer}:{v[0]:.2f}" for layer, v in enumerate(a["transfer"]) if layer))
            (out_dir / "results.json").write_text(json.dumps(results, default=float))   # incremental
        if not cell_tests[mode]:
            continue
        base = timed(f"{mode} baseline", lambda: runner.score(prompts))
        contrast = S.contrast(base, regimes)
        results["contrast"][mode] = contrast
        # per prompt, so results can be split afterwards (e.g. forced vs natural think closes)
        results.setdefault("base", {})[mode] = base.tolist()
        results.setdefault("forced", {})[mode] = ([bool(f) for f in post["forced"]] if mode == "on"
                                                  else [False] * len(rows))
        for key, cell in cells.items():
            if cell["mode"] != mode:
                continue
            res = results["cells"][key] = {}
            if "ablate" in cell_tests[mode]:
                res["ablate"], res["ablate_per_prompt"] = timed(
                    f"{key} ablate", lambda: run_ablate(runner, prompts, cell, regimes, base))
                # saved before steering, so a stopped run keeps the ablation
                (out_dir / "results.json").write_text(json.dumps(results, default=float))
            if "steer" in cell_tests[mode]:
                res["steer"] = timed(f"{key} steer",
                                     lambda: run_steer(runner, prompts, cell, regimes, log_h, base, contrast))
            res["verdicts"] = verdicts(mode, res)
            print_cell(key, res, contrast)
            (out_dir / "results.json").write_text(json.dumps(results, default=float))   # incremental
    (out_dir / "results.json").write_text(json.dumps(results, default=float))
    print(f"\nwrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--inputs", required=True,
                        help="steering/inputs_<model>_<name>.npz from analysis/steering_prepare.py")
    parser.add_argument("--tests", default=",".join(TESTS), help=f"comma-separated subset of {','.join(TESTS)}")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N near/both prompts (smoke)")
    parser.add_argument("--batch-size", type=int, default=32, help="batch for thinking-off prompts (~200 tokens)")
    parser.add_argument("--long-batch-size", type=int, default=8, help="batch for cached reasoning (~2.5k tokens)")
    parser.add_argument("--out", default=None, help="output base (default: out/ or $WORKSPACE/results/horizon)")
    main(parser.parse_args())
