"""E2: paraphrase transfer (plan row R1).

Six constraint wordings (starter + 5). For each, extract at E0's probe layer
+/- 2, train the horizon probe on template i, evaluate on template j. Writes a
transfer matrix per layer and one heatmap for the best layer.

    python experiments/exp2_paraphrase.py --model Qwen/Qwen3-8B --thinking off
"""

import json

import numpy as np

from common import parse_args
from spar_horizon.extract import extract
from spar_horizon.geometry import transfer_matrix
from spar_horizon.model_io import load_model
from spar_horizon.plots import heatmap
from spar_horizon.prompts import CONSTRAINT_TEMPLATES, build_prompts
from spar_horizon.runs import Manifest, resolve_out_dir

EXP = "exp2"


def probe_layers_from_e0(cfg, out, n_layers):
    m = resolve_out_dir("exp0", cfg, out) / "manifest.json"
    if not m.exists():
        print("no E0 manifest: extracting at every layer")
        return list(range(n_layers)), None
    d = json.loads(m.read_text())
    c = d["probe_layer"]
    return [l for l in range(c - 2, c + 3) if 0 <= l < n_layers], d["probe_pos"]


def run(cfg, out=None):
    out_dir = resolve_out_dir(EXP, cfg, out)
    manifest = Manifest(EXP, cfg, out_dir)
    tokenizer, model, device = load_model(cfg)
    n_layers = len(model.model.layers) + 1
    layers, probe_pos = probe_layers_from_e0(cfg, out, n_layers)
    names = list(CONSTRAINT_TEMPLATES)
    banks, log_hs, kept_counts = {l: [] for l in layers}, [], {}
    for name in names:
        records = [r for r in build_prompts(seed=cfg.seed, limit=cfg.limit, template=name)
                   if r["horizon"] is not None]
        print(f"\n[{name}] {len(records)} prompts")
        ext = extract(tokenizer, model, device, [r["prompt"] for r in records], cfg, layers=layers)
        lh = np.log10([r["horizon"] for r, k in zip(records, ext.kept) if k])
        pos = probe_pos if probe_pos is not None else ext.n_suffix - 1
        pos = min(pos, ext.activations[0].shape[1] - 1)
        for li, l in enumerate(layers):
            banks[l].append(ext.activations[li][:, pos])
        log_hs.append(lh)
        kept_counts[name] = int(ext.kept.sum())

    results, best = {}, (None, -1)
    for l in layers:
        M = transfer_matrix(banks[l], log_hs)
        off = M[~np.eye(len(names), dtype=bool)].mean()
        results[l] = {"matrix": M.tolist(), "diag_mean": float(np.diag(M).mean()),
                      "offdiag_mean": float(off)}
        np.savetxt(out_dir / f"transfer_L{l}.csv", M, delimiter=",", header=",".join(names))
        print(f"layer {l}: diag {np.diag(M).mean():.3f}  off-diag {off:.3f}")
        if off > best[1]:
            best = (l, off)
    M = np.array(results[best[0]]["matrix"])
    heatmap(M, names, names, out_dir / "transfer_best.png",
            f"probe transfer |rho|, layer {best[0]} (train rows, test cols)")
    manifest.add(layers=layers, probe_pos=probe_pos, templates=names, kept=kept_counts,
                 per_layer=results, best_layer=best[0], best_offdiag=best[1])
    manifest.write()
    return out_dir


if __name__ == "__main__":
    args, cfg = parse_args(__doc__)
    run(cfg, args.out)
