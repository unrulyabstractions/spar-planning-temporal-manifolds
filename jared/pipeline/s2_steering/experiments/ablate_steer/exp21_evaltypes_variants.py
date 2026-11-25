"""E21: cheap variants of E19's answer-slot steering on the eval types (EvalsPromptsV2.md §9).

    python -m pipeline.s2_steering.experiments.ablate_steer.exp21_evaltypes_variants \\
        --inputs steering/inputs_Qwen3-8B_evaltypes.npz --out /workspace/results/horizon
    python -m pipeline.s2_steering.experiments.ablate_steer.exp21_evaltypes_variants \\
        --inputs steering/inputs_Qwen3-8B_evaltypes.npz --limit 2 --out /workspace/results/horizon/smoke   # smoke

Same rows, sequences and edit site as E19 (the pre-label ':' at the last position; thinking off with
"I choose:" prefilled, thinking on with E0's own reasoning cut before the label), so every condition is
a one-token pass on one prefix cache per batch. Conditions, each with the bank's fit at its layer:
1. dose: the curve at L24, +-1, +-2, +-4 decades; random 0 at +-2 and +-4 as the damage control;
2. direction at L24, +-2: mean difference (CAA), the probe line, random 1 (curve, random 0 from 1);
3. layer: the curve and random 0 at +-2 at each other layer of the sweep;
4. stacked: the curve at +-2 at L22, L24, L26 and L28 at once;
5. removal at L24: mean-ablate the top-k horizon subspace, k = 1, 2, 4, 8, and three random
   subspaces at each k.
Outcome: near log-odds. Writes <out>/exp21_<inputs name>/results.json (per row: sample_uid, base and
every condition; per mode: the curve range of each cell), saved after each mode.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from pipeline.s2_steering.experiments.ablate_steer.exp8_steering import delta_rows
from pipeline.s2_steering.experiments.ablate_steer.exp19_evaltypes_steer import limited, load
from spar_horizon import steering as S
from spar_horizon.config import Settings
from spar_horizon.model_io import load_model
from spar_horizon.runs import code_hash, out_base, stamp

LAYER = 24
SWEEP = (16, 20, 22, 26, 28, 32)
STACK = (22, 24, 26, 28)
ABLATE_KS = (1, 2, 4, 8)
N_RANDOM_SUBSPACES = 3
CHUNK = 32          # rows per progress step


def steer_conditions():
    """(name, layers, direction, alpha): additive steps, each layer with its own cell."""
    out = [(f"L{LAYER} curve {a:+d}", (LAYER,), "curve", a) for a in (-4, -2, -1, 1, 2, 4)]
    out += [(f"L{LAYER} random 0 {a:+d}", (LAYER,), "random 0", a) for a in (-4, -2, 2, 4)]
    out += [(f"L{LAYER} {d} {a:+d}", (LAYER,), d, a) for d in ("mean difference", "line", "random 1") for a in (-2, 2)]
    out += [(f"L{layer} {d} {a:+d}", (layer,), d, a) for layer in SWEEP for d in ("curve", "random 0") for a in (-2, 2)]
    out += [(f"L{'+'.join(map(str, STACK))} curve {a:+d}", STACK, "curve", a) for a in (-2, 2)]
    return out


def ablate_conditions(d, basis):
    """(name, orthonormal basis [k, d]) at L24."""
    out = []
    for k in ABLATE_KS:
        out.append((f"L{LAYER} ablate horizon k{k}", basis[:k]))
        out += [(f"L{LAYER} ablate random {s} k{k}", S.random_orthonormal(d, k, seed=s))
                for s in range(N_RANDOM_SUBSPACES)]
    return out


def edits_for(cells, mode, log_h, device):
    """Names and score_last_edits specs: the unedited base first, then every condition."""
    names, specs = ["base"], [None]
    for name, layers, direction, alpha in steer_conditions():
        deltas = [torch.tensor(delta_rows(cells[mode, layer], direction, alpha, log_h), device=device,
                               dtype=torch.float32) for layer in layers]
        names.append(name)
        specs.append((layers, lambda rows, ds=deltas: tuple(S.Add(dl[rows]) for dl in ds)))
    cell = cells[mode, LAYER]
    mean = torch.tensor(cell["mean"], device=device, dtype=torch.float32)
    for name, basis in ablate_conditions(len(cell["mean"]), cell["basis"]):
        edit = S.MeanAblate(mean, torch.tensor(basis, device=device, dtype=torch.float32))
        names.append(name)
        specs.append((LAYER, lambda rows, e=edit: e))
    return names, specs


def run_mode(model, device, pad_id, mode, cells, meta, ids, labels, rows, batch_size):
    records = [meta["rows"][i] for i in rows]
    near_first = [r["near_label"] == "a" for r in records]
    log_h = np.log10([r["horizon"] for r in records])
    names = edits_for(cells, mode, log_h[:1], device)[0]
    stamp(f"thinking {mode}: {len(ids)} rows x {len(names)} conditions")
    scored = np.zeros((len(names), len(ids)))
    # chunks of rows only for the progress bar; each chunk's steps are built for its own rows
    for start in tqdm(range(0, len(ids), CHUNK), desc=f"thinking {mode}", unit="chunk"):
        part = slice(start, start + CHUNK)
        _, specs = edits_for(cells, mode, log_h[part], device)
        scored[:, part] = S.score_last_edits(model, device, ids[part], near_first[part], labels[part], pad_id,
                                             batch_size, specs)
    return {"rows": [{"sample_uid": r["sample_uid"], **{n: float(scored[c, j]) for c, n in enumerate(names)}}
                     for j, r in enumerate(records)],
            "curve_range": {str(layer): [float(cells[mode, layer]["curve_logh"].min()),
                                         float(cells[mode, layer]["curve_logh"].max())]
                            for m, layer in cells if m == mode}}


def main(args):
    meta, cells, sequences = load(args.inputs)
    if args.limit:
        sequences = limited(meta, sequences, args.limit)
    cfg = Settings(model=meta["model"], thinking="off", prefill="I choose:", bank="full")
    tokenizer, model, device = load_model(cfg)
    out_dir = out_base(args.out) / Path(args.inputs).stem.replace("inputs_", "exp21_", 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {"inputs": str(args.inputs), "code_hash": code_hash(), "modes": {}, "phases": {}}
    for mode in args.modes.split(","):
        ids, labels, rows = sequences[mode]
        t0 = time.time()
        batch = args.batch_size if mode == "off" else args.long_batch_size
        results["modes"][mode] = run_mode(model, device, tokenizer.pad_token_id, mode, cells, meta, ids, labels,
                                          rows, batch)
        results["phases"][mode] = round(time.time() - t0, 1)
        stamp(f"thinking {mode} done, {results['phases'][mode]} s")
        (out_dir / "results.json").write_text(json.dumps(results))   # incremental
    print(f"wrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--inputs", required=True)
    p.add_argument("--modes", default="off,on")
    p.add_argument("--limit", type=int, default=None, help="rows per type and variant (smoke)")
    p.add_argument("--batch-size", type=int, default=32, help="batch for thinking-off prompts")
    p.add_argument("--long-batch-size", type=int, default=2, help="batch for cached reasoning (2 fits a 24 GB card)")
    p.add_argument("--out", default=None)
    main(p.parse_args())
