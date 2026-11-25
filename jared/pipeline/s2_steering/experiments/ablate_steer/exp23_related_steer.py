"""E23: does the horizon edit move power seeking, survival instinct, wealth seeking and reward hacking
more than random directions of the same length do? (EvalsRelatedQualities.md §3)

    python -m pipeline.s2_steering.experiments.ablate_steer.exp23_related_steer \\
        --inputs steering/related_box/inputs_Qwen3-8B_related.npz \\
        --eval-probe $WORKSPACE/results/horizon/eval_probe_Qwen3-8B.npz --out $WORKSPACE/results/horizon
    ... --limit 4 --k 2 --out $WORKSPACE/results/horizon/smoke                                     # smoke

Inputs from pipeline/s1_data_vectors/analysis/related_steering_prepare.py. Every row (fit and eval
split) is E0's reasoning cut before the label. Edits at its last position (the pre-label ':') at
--layer, all sharing one prefix pass per batch (score_last_edits):
- none;
- the curve +-2 decades from the row's start horizon;
- K random directions (E8's "random <seed>", seeds 0..K-1), each step as long as the row's curve step, +-2;
- the row's set behaviour CAA, matched length, +-2, when the set has one (positive control).
Outcome: the behaviour log-odds, logit(matching label) - logit(other). With --eval-probe, the
eval-awareness probe's logit at --probe-layers, read in the same passes. Writes
<out>/exp23_<inputs name>/results.json (per set: the conditions and per row the outcome and probe
readouts under each), saved after each set.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pipeline.s1_data_vectors.analysis.related_steering_prepare import load_inputs
from pipeline.s2_steering.experiments.ablate_steer.exp8_steering import adder, delta_rows
from spar_horizon import steering as S
from spar_horizon.config import Settings
from spar_horizon.model_io import load_model
from spar_horizon.runs import code_hash, out_base, stamp

ALPHAS = (-2, 2)
ROW_FIELDS = ("sample_uid", "set_id", "variant", "twin_id", "split", "cheat_method")


def conditions(k, has_caa):
    directions = ["curve", *(f"random {seed}" for seed in range(k))] + (["caa"] if has_caa else [])
    return [(direction, alpha) for direction in directions for alpha in ALPHAS]


def deltas(cell, caa, direction, alpha, log_h):
    if direction == "caa":
        return S.matched_deltas(cell, caa, alpha, log_h)
    return delta_rows(cell, direction, alpha, log_h)


def load_probes(path, layers):
    """[(layer, mean, coef, intercept)] of the eval-awareness probe at `layers`, and whether it passed."""
    z = np.load(path)
    saved = z["layers"].tolist()
    missing = [layer for layer in layers if layer not in saved]
    if missing:
        raise ValueError(f"{path} has no probe at layers {missing} (it has {saved})")
    probes = [(layer, z["mean"][saved.index(layer)], z["coef"][saved.index(layer)],
               float(z["intercept"][saved.index(layer)])) for layer in layers]
    return probes, bool(z["passed"])


def limited(rows, limit):
    """Indices of the first `limit` rows of each set and variant (smoke and pilot)."""
    seen, keep = {}, []
    for i, r in enumerate(rows):
        key = (r["set_id"], r["variant"])
        seen[key] = seen.get(key, 0) + 1
        if limit is None or seen[key] <= limit:
            keep.append(i)
    return keep


def run_set(model, device, pad_id, inputs, idx, caa, layer, k, probes, batch_size):
    """Per-row results for the rows `idx` of one set, and the condition names."""
    cell, log_h = inputs["cell"], inputs["log_h"][idx]
    conds = conditions(k, caa is not None)
    edits = [None] + [(layer, adder(deltas(cell, caa, direction, alpha, log_h), device))
                      for direction, alpha in conds]
    scored = S.score_last_edits(model, device, [inputs["ids"][i] for i in idx],
                                [inputs["near_first"][i] for i in idx], [inputs["labels"][i] for i in idx],
                                pad_id, batch_size, edits, probes=probes)
    lo, readouts = scored if probes else (scored, None)
    names = ["base"] + [f"{direction} {alpha:+d}" for direction, alpha in conds]
    out = []
    for j, i in enumerate(idx):
        r = inputs["meta"]["rows"][i]
        row = {**{key: r[key] for key in ROW_FIELDS if key in r}, "log_h": float(log_h[j]),
               **{name: float(lo[e, j]) for e, name in enumerate(names)}}
        if probes:
            row["probe"] = {f"L{p_layer}": {name: float(readouts[e, j, p]) for e, name in enumerate(names)}
                            for p, (p_layer, *_) in enumerate(probes)}
        out.append(row)
    return out, names[1:]


def main(args):
    inputs = load_inputs(args.inputs)
    meta = inputs["meta"]
    if args.layer != meta["layer"]:
        raise ValueError(f"--layer {args.layer}, but the inputs' cell and start horizons are at L{meta['layer']}")
    probes, passed = (load_probes(args.eval_probe, [int(x) for x in args.probe_layers.split(",")])
                      if args.eval_probe else (None, None))
    tokenizer, model, device = load_model(Settings(model=meta["model"]))
    out_dir = out_base(args.out) / Path(args.inputs).stem.replace("inputs_", "exp23_", 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {"inputs": str(args.inputs), "code_hash": code_hash(), "layer": args.layer, "k": args.k,
               "limit": args.limit, "alphas": list(ALPHAS),
               "probe": None if probes is None else {"file": str(args.eval_probe), "passed": passed,
                                                     "layers": [p[0] for p in probes]},
               "sets": {}}
    keep = limited(meta["rows"], args.limit)
    for set_id in dict.fromkeys(meta["rows"][i]["set_id"] for i in keep):
        idx = [i for i in keep if meta["rows"][i]["set_id"] == set_id]
        caa = inputs["caa"].get(set_id)
        t0 = time.time()
        stamp(f"{set_id}: {len(idx)} rows, {1 + len(conditions(args.k, caa is not None))} edits per row ...")
        rows, names = run_set(model, device, tokenizer.pad_token_id, inputs, idx, caa, args.layer, args.k,
                              probes, args.batch_size)
        seconds = round(time.time() - t0, 1)
        results["sets"][set_id] = {"conditions": names, "seconds": seconds, "rows": rows}
        stamp(f"{set_id} done, {seconds} s ({seconds / len(idx):.2f} s per row)")
        (out_dir / "results.json").write_text(json.dumps(results))   # incremental
    print(f"wrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--inputs", required=True)
    p.add_argument("--eval-probe", default=None, help="eval_probe_<model>.npz from eval_probe_fit.py")
    p.add_argument("--probe-layers", default="28,32")
    p.add_argument("--layer", type=int, default=24, help="the edited layer; must be the inputs' cell layer")
    p.add_argument("--k", type=int, default=19, help="random directions")
    p.add_argument("--limit", type=int, default=None, help="rows per set and variant (smoke, pilot)")
    # 8 traces of ~3k tokens ran out of memory on a 4090 (E20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--out", default=None)
    main(p.parse_args())
