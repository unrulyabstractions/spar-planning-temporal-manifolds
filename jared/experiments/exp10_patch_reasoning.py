"""E10: patch the horizon phrase before reasoning, then let the model reason
(spec: 07_spec_steering.md section 12). Thinking on, full bank.

Each pair is a near/both prompt (recipient) and the same prompt with a horizon
from the other regime (donor), equal in token length. The recipient's hidden
states at the horizon phrase are overwritten at one layer with the donor's (all
of them, or only their top-8 horizon-subspace component), the model generates
its reasoning and answer greedily, and the choice is compared with the unedited
recipient and with the donor prompt itself.

    python experiments/exp10_patch_reasoning.py --out /workspace/results/horizon
    python experiments/exp10_patch_reasoning.py --model Qwen/Qwen3.5-0.8B --pairs 4 --self-pairs 2 \\
        --max-think-tokens 32 --batch-size 4 --out out          # local smoke (plumbing only)

Writes <out>/exp10_<model>_reasoning/: subspace.npz (the fitted bases),
conditions/<name>.json and .npz (per pair: near log-odds, label, forced, think
length, which horizon the reasoning names; the generated ids), summary.json.
Each condition is saved when it finishes and skipped on a rerun, so a stopped
run resumes where it stopped. Several copies (one per GPU, see parallel.sh) share
the queue: a condition is claimed by creating conditions/<name>.lock, so a
crashed run leaves locks that parallel.sh clears before starting.
"""

import argparse
import json
import random
import re
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from spar_horizon import steering as S  # noqa: E402
from spar_horizon.banks import build_records  # noqa: E402
from spar_horizon.config import Settings  # noqa: E402
from spar_horizon.full_bank import render_variant  # noqa: E402
from spar_horizon.model_io import _free_memory, _generate_rows, chat_text, label_ids, load_model  # noqa: E402
from spar_horizon.runs import code_hash, out_base, stamp  # noqa: E402

K = 8                     # horizon-subspace dimensions swapped by the subspace patch
MIN_PAIR_GAP = 1.0        # recipient/donor-text log-odds gap below which a pair carries no transfer signal


def conditions(layers):
    """(name, layer, positions, edit) in priority order; positions "phrase" or "last",
    edit "full", "subspace", "random" or "self"."""
    out = [("recipient", None, None, None), ("donor text", None, None, None)]
    for layer in layers:
        out += [(f"full phrase L{layer}", layer, "phrase", "full"),
                (f"subspace last L{layer}", layer, "last", "subspace"),
                (f"full last L{layer}", layer, "last", "full"),
                (f"random last L{layer}", layer, "last", "random")]
    out.append((f"self last L{layers[0]}", layers[0], "last", "self"))
    return out


def build_pairs(records, tokenizer, cfg, n_pairs):
    """Crossing pairs (steering.crossing_pairs, seed 0) with the horizon phrase
    positions: the union of the recipient's and the donor's phrase tokens."""
    selected = [r for r in records if r["regime"] in ("near", "both")]
    pairs = S.crossing_pairs(selected, tokenizer, cfg, random.Random(0), n_pairs)
    for p in pairs:
        rec = selected[p["record"]]
        donor_rec = {**rec, "horizon_text": p["donor_horizon"]}
        p["rec"], p["donor_prompt"] = rec, render_variant(donor_rec)
        phrase = set(S.phrase_positions(tokenizer, chat_text(tokenizer, rec["prompt"], cfg), rec["horizon_text"]))
        phrase |= set(S.phrase_positions(tokenizer, chat_text(tokenizer, p["donor_prompt"], cfg), p["donor_horizon"]))
        p["phrase"] = sorted(phrase)
    return pairs


@torch.no_grad()
def states_at(model, device, tokenizer, id_lists, positions, layers, batch_size):
    """{layer: list of [len(positions[i]), d] float32 CPU tensors}: hidden_states[layer]
    of each prompt (forward pass over the prompt only) at its positions."""
    out = {layer: [] for layer in layers}
    for start in range(0, len(id_lists), batch_size):
        rows = slice(start, start + batch_size)
        _, hidden, offsets = S.forward(model, device, id_lists[rows], tokenizer.pad_token_id, hidden=True)
        for layer in layers:
            values = S.gather(hidden[layer], positions[rows], offsets).cpu()
            sizes = [len(p) for p in positions[rows]]
            out[layer] += list(torch.split(values, sizes))
    return out


def fit_subspaces(model, device, tokenizer, cfg, records, layers, batch_size):
    """Top-K horizon-average PCs at the last horizon-phrase token, per layer, from
    one prompt-only pass over every prompt with a horizon (as in the saved E0
    activations, but at a position E0 did not keep)."""
    rows = [r for r in records if r["horizon"] is not None]
    ids = [S.encode(tokenizer, r["prompt"], cfg) for r in rows]
    last = [[S.phrase_positions(tokenizer, chat_text(tokenizer, r["prompt"], cfg), r["horizon_text"])[-1]]
            for r in rows]
    states = states_at(model, device, tokenizer, ids, last, layers, batch_size)
    log_h = np.log10([r["horizon"] for r in rows])
    return {layer: S.fit_cell(torch.cat(states[layer]).numpy().astype(np.float64), log_h, k_max=K)["basis"]
            for layer in layers}


def generate(model, device, tokenizer, cfg, id_lists, batch_size, patch=None):
    """Greedy reasoning and answer per sequence (model_io._generate_rows, no cache).
    `patch(rows)` returns the prompt_patch block for one batch, or None. On out of
    memory the batch is retried at two thirds of the size."""
    results = [None] * len(id_lists)
    done = 0
    bar = tqdm(total=len(id_lists), desc="generate", unit="prompt", mininterval=5, dynamic_ncols=True)
    while done < len(id_lists):
        idx = list(range(done, min(done + batch_size, len(id_lists))))
        try:
            with patch(idx) if patch else nullcontext():
                _generate_rows(tokenizer, model, device, id_lists, idx, cfg, results)
        except torch.OutOfMemoryError:
            if batch_size == 1:
                raise
            _free_memory(device)
            batch_size = max(1, batch_size * 2 // 3)
            print(f"out of memory; continuing at batch size {batch_size}", flush=True)
            continue
        done += len(idx)
        bar.update(len(idx))
    bar.close()
    return results


@torch.no_grad()
def score(model, device, tokenizer, cfg, generations, near_first, batch_size, patch=None):
    """Near log-odds at the token before each generated label, with the same patch
    applied to the prompt, and the label as written."""
    with_space, without_space = label_ids(tokenizer, cfg=cfg), label_ids(tokenizer)
    seqs = [g.ids[:g.answer_start].tolist() for g in generations]
    written = [tokenizer.decode([int(g.ids[g.answer_start])]) for g in generations]
    labels = [with_space if w.startswith(" ") else without_space for w in written]
    out = []
    for start in range(0, len(seqs), batch_size):
        idx = list(range(start, min(start + batch_size, len(seqs))))
        with patch(idx) if patch else nullcontext():
            logits, _, _ = S.forward(model, device, [seqs[i] for i in idx], tokenizer.pad_token_id)
        out.append(S.near_log_odds(logits, [near_first[i] for i in idx], [labels[i] for i in idx]))
    return np.concatenate(out), [w.strip() for w in written]


def names_horizon(think, horizon_text):
    """Whether the reasoning writes this horizon ("1 year" does not match "11 years")."""
    return re.search(rf"(?<![\d.,]){re.escape(horizon_text)}", think) is not None


def run_condition(name, layer, where, edit, pairs, ctx):
    """Generate and score one condition; returns the per-pair record and the ids."""
    model, device, tokenizer, cfg = ctx["model"], ctx["device"], ctx["tokenizer"], ctx["cfg"]
    pairs = pairs[:ctx["self_pairs"]] if edit == "self" else pairs
    ids = [p["donor"] if name == "donor text" else p["recipient"] for p in pairs]
    patch = None
    if layer is not None:
        positions = [p["phrase"] if where == "phrase" else p["phrase"][-1:] for p in pairs]
        source = [p["recipient"] if edit == "self" else p["donor"] for p in pairs]
        donor = states_at(model, device, tokenizer, source, positions, [layer], ctx["batch_size"])[layer]
        basis = {"subspace": ctx["bases"].get(layer), "random": ctx["random"]}.get(edit)
        basis = None if basis is None else torch.tensor(basis, dtype=torch.float32)

        def patch(idx):
            return S.prompt_patch(model, layer, [positions[i] for i in idx], [donor[i] for i in idx], basis)

    generations = generate(model, device, tokenizer, cfg, ids, ctx["batch_size"], patch)
    near_first = [p["rec"]["near_label"] == "a" for p in pairs]
    near_lo, labels = score(model, device, tokenizer, cfg, generations, near_first, ctx["score_batch_size"], patch)
    rows = []
    for p, g, lo, label in zip(pairs, generations, near_lo, labels):
        think = g.think_text(tokenizer)
        rows.append({"sample_uid": p["rec"]["sample_uid"], "regime": p["rec"]["regime"],
                     "recipient_horizon": p["rec"]["horizon_text"], "donor_horizon": p["donor_horizon"],
                     "near_label": p["rec"]["near_label"], "near_lo": float(lo), "label": label,
                     "forced": g.forced, "think_len": g.think_len,
                     "names_recipient": names_horizon(think, p["rec"]["horizon_text"]),
                     "names_donor": names_horizon(think, p["donor_horizon"])})
    lengths = [len(g.ids) for g in generations]
    arrays = {"ids": np.concatenate([g.ids.cpu().numpy() for g in generations]).astype(np.int32),
              "offsets": np.r_[0, np.cumsum(lengths)], "answer_start": [g.answer_start for g in generations],
              "prompt_len": [g.prompt_len for g in generations]}
    return {"name": name, "layer": layer, "positions": where, "edit": edit, "rows": rows}, arrays


def summarize(results):
    """Transfer toward the donor text per condition (pairs with a gap above
    MIN_PAIR_GAP), label agreement with the donor text, mention rates, and the
    spec's criteria."""
    base = np.array([r["near_lo"] for r in results["recipient"]["rows"]])
    target = np.array([r["near_lo"] for r in results["donor text"]["rows"]])
    target_label = [r["label"] for r in results["donor text"]["rows"]]
    gap = target - base
    out = {}
    for name, res in results.items():
        rows = res["rows"]
        n = len(rows)
        lo = np.array([r["near_lo"] for r in rows])
        transfer = (lo - base[:n]) / gap[:n]
        used = np.abs(gap[:n]) > MIN_PAIR_GAP
        out[name] = {"n": n, "n_used": int(used.sum()),
                     "transfer": S.bootstrap_mean(transfer[used]),
                     "transfer_values": transfer.tolist(),
                     "label_as_donor_text": float(np.mean([r["label"] == t for r, t in zip(rows, target_label)])),
                     "label_changed": int(sum(r["label"] != b["label"] for r, b in zip(rows, results["recipient"]["rows"]))),
                     "names_recipient": float(np.mean([r["names_recipient"] for r in rows])),
                     "names_donor": float(np.mean([r["names_donor"] for r in rows])),
                     "forced": float(np.mean([r["forced"] for r in rows]))}
    out["criteria"] = {}
    for name in results:
        if not name.startswith("subspace last"):
            continue
        layer = name.split()[-1]
        full, rand = f"full last {layer}", f"random last {layer}"
        if full not in results or rand not in results:
            continue
        used = np.abs(gap) > MIN_PAIR_GAP
        sub_t = np.array(out[name]["transfer_values"])[used]
        diff = S.bootstrap_mean(sub_t - np.array(out[rand]["transfer_values"])[used])
        out["criteria"][f"subspace read {layer}"] = {
            "subspace": out[name]["transfer"][0], "full_last": out[full]["transfer"][0],
            "random": out[rand]["transfer"][0], "minus_random_low": diff[1],
            "pass": bool(out[name]["transfer"][0] >= 0.5 * out[full]["transfer"][0] and diff[1] > 0
                         and out[rand]["transfer"][0] <= 0.1)}
    self_name = next((n for n in results if n.startswith("self")), None)
    if self_name:
        out["criteria"]["plumbing"] = {"labels_changed": out[self_name]["label_changed"],
                                       "pass": out[self_name]["label_changed"] <= 2}
    return out


def report(out_dir, name=None):
    """Summary of every condition finished so far, by any worker (summary.json);
    prints one condition's line, or the criteria when `name` is None."""
    results = {}
    for file in sorted((out_dir / "conditions").glob("*.json")):
        result = json.loads(file.read_text())
        results[result["name"]] = result
    if "recipient" not in results or "donor text" not in results:
        return
    summary = summarize(results)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    if name is not None:
        s = summary[name]
        print(f"  transfer {s['transfer'][0]:+.2f} [{s['transfer'][1]:+.2f}, {s['transfer'][2]:+.2f}] "
              f"({s['n_used']} pairs), label as donor text {s['label_as_donor_text']:.2f}, "
              f"names donor {s['names_donor']:.2f} / recipient {s['names_recipient']:.2f}, "
              f"forced {s['forced']:.2f}", flush=True)
        return
    for criterion, verdict in summary["criteria"].items():
        print(f"{criterion:20} {'PASS' if verdict['pass'] else 'fail'}  "
              + ", ".join(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}"
                          for k, v in verdict.items() if k != "pass"))


def main(args):
    cfg = Settings(model=args.model, thinking="on", prefill="", choice_prefix="I choose:", bank="full",
                   n_prompts=500, seed=0, max_think_tokens=args.max_think_tokens, batch_size=args.batch_size)
    out_dir = out_base(args.out) / f"exp10_{cfg.model.split('/')[-1]}_reasoning"
    (out_dir / "conditions").mkdir(parents=True, exist_ok=True)
    records = build_records(cfg)
    tokenizer, model, device = load_model(cfg)
    layers = [int(x) for x in args.layers.split(",")]
    pairs = build_pairs(records, tokenizer, cfg, args.pairs)
    stamp(f"{len(pairs)} pairs ({sum(p['rec']['regime'] == 'near' for p in pairs)} near recipients), "
          f"phrase of 1-{max(len(p['phrase']) for p in pairs)} tokens, layers {layers}")
    t0 = time.time()
    path = out_dir / "subspace.npz"
    if path.exists():
        z = np.load(path)
        bases = {layer: z[f"L{layer}"] for layer in layers}
    else:
        bases = fit_subspaces(model, device, tokenizer, cfg, records, layers, args.score_batch_size)
        # write then rename, so a worker starting meanwhile never reads half a file
        np.savez(out_dir / "subspace.tmp.npz", **{f"L{layer}": b for layer, b in bases.items()})
        (out_dir / "subspace.tmp.npz").replace(path)
    d = model.config.hidden_size
    ctx = {"model": model, "device": device, "tokenizer": tokenizer, "cfg": cfg, "bases": bases,
           "random": S.random_orthonormal(d, K, seed=0), "batch_size": args.batch_size,
           "score_batch_size": args.score_batch_size, "self_pairs": args.self_pairs}
    stamp(f"subspaces ready ({time.time() - t0:.0f} s)")
    if args.fit_only:
        return
    (out_dir / "manifest.json").write_text(json.dumps({
        "settings": cfg.to_dict(), "layers": layers, "pairs": len(pairs), "self_pairs": args.self_pairs,
        "code_hash": code_hash()}, indent=1))
    todo = conditions(layers)
    if args.only:
        todo = [c for c in todo if c[0] in args.only.split(";")]
    for name, layer, where, edit in todo:
        file = out_dir / "conditions" / f"{name.replace(' ', '_')}.json"
        if file.exists():
            continue
        # several workers (one per GPU) share the queue: the first to create the lock runs the condition
        try:
            file.with_suffix(".lock").mkdir()
        except FileExistsError:
            continue
        t0 = time.time()
        stamp(f"{name} ...")
        result, arrays = run_condition(name, layer, where, edit, pairs, ctx)
        result["seconds"] = round(time.time() - t0, 1)
        np.savez(file.with_suffix(".npz"), **arrays)
        # write then rename: other workers read every finished condition for the summary
        file.with_suffix(".tmp").write_text(json.dumps(result))
        file.with_suffix(".tmp").replace(file)
        stamp(f"{name} done in {result['seconds']} s")
        report(out_dir, name)
    report(out_dir)
    stamp(f"wrote {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--layers", default="16,12", help="patch layers, in priority order")
    parser.add_argument("--pairs", type=int, default=96)
    parser.add_argument("--self-pairs", type=int, default=24, help="pairs in the self-patch plumbing control")
    parser.add_argument("--max-think-tokens", type=int, default=3072)
    parser.add_argument("--batch-size", type=int, default=12, help="generation batch (E0's on the 4090)")
    parser.add_argument("--score-batch-size", type=int, default=8, help="batch for prompt and scoring passes")
    parser.add_argument("--fit-only", action="store_true",
                        help="fit and save the subspaces, then stop (parallel.sh runs this once before the workers)")
    parser.add_argument("--only", default=None, help='";"-separated condition names to run (default: all)')
    parser.add_argument("--out", default=None, help="output base (default: out/ or $WORKSPACE/results/horizon)")
    main(parser.parse_args())
