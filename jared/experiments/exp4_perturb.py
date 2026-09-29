"""E4: prompt perturbation (Jared's addition).

Variants of the base bank: distractor sentence, horizon sentence first, and
an implicit-horizon bank where a life event implies the duration. For each:
probe transfer from the clean bank (train on clean, test on variant) and the
order/label stability tests.

    python experiments/exp4_perturb.py --model Qwen/Qwen3-8B --thinking off
"""

import json

import numpy as np
from scipy.stats import spearmanr

from common import parse_args
from spar_horizon.behavior import behavior
from spar_horizon.extract import extract
from spar_horizon.geometry import horizon_probe
from spar_horizon.model_io import load_model
from spar_horizon.plots import heatmap
from spar_horizon.prompts import (IMPLICIT_HORIZONS, PERTURBATIONS, build_implicit_prompts,
                                  build_prompts, choice_prompt, subset)
from spar_horizon.runs import Manifest, resolve_out_dir

EXP = "exp4"


def variants(cfg):
    """name -> (records, prompt_fn(rec, **kw))."""
    base = [r for r in build_prompts(seed=cfg.seed, limit=cfg.limit) if r["horizon"] is not None]
    out = {"clean": (base, lambda r, **kw: choice_prompt(r["rewards"], r["delays"], r["horizon"], **kw))}
    for name, fn in PERTURBATIONS.items():
        out[name] = (base, lambda r, fn=fn, **kw: fn(choice_prompt(r["rewards"], r["delays"], r["horizon"], **kw)))
    implicit = subset(build_implicit_prompts(), cfg.limit)
    out["implicit"] = (implicit, lambda r, **kw: choice_prompt(
        r["rewards"], r["delays"], r["horizon"], constraint=IMPLICIT_HORIZONS[r["horizon"]], **kw))
    return out


def run(cfg, out=None):
    out_dir = resolve_out_dir(EXP, cfg, out)
    manifest = Manifest(EXP, cfg, out_dir)
    tokenizer, model, device = load_model(cfg)
    m0 = resolve_out_dir("exp0", cfg, out) / "manifest.json"
    layer, pos = (lambda d: (d["probe_layer"], d["probe_pos"]))(json.loads(m0.read_text())) \
        if m0.exists() else (None, None)
    rows, names, probe = [], [], None
    for name, (records, prompt_fn) in variants(cfg).items():
        print(f"\n[{name}] {len(records)} prompts")
        prompts = [prompt_fn(r) for r in records]
        ext = extract(tokenizer, model, device, prompts, cfg, layers=[layer] if layer is not None else None)
        recs = [r for r, k in zip(records, ext.kept) if k]
        log_h = np.log10([r["horizon"] for r in recs])
        if layer is None:  # no E0: pick the layer on the clean bank
            p = ext.n_suffix - 1
            layer = int(np.nanargmax([horizon_probe(a[:, p], log_h)[0] for a in ext.activations]))
            pos = p
            acts = ext.activations[layer]
        else:
            acts = ext.activations[0]
        X = acts[:, min(pos, acts.shape[1] - 1)]
        if name == "clean":
            r2, _, probe = horizon_probe(X, log_h)
            transfer = float(abs(spearmanr(probe.predict(X), log_h)[0]))
        else:
            transfer = float(abs(spearmanr(probe.predict(X), log_h)[0]))
        self_r2 = horizon_probe(X, log_h)[0] if len(X) >= 5 else float("nan")
        beh = behavior(tokenizer, model, device, recs, [a for a in ext.answers if a is not None],
                       cfg, prompt_fn=prompt_fn)
        rows.append([transfer, self_r2, beh["order"][0] / max(beh["order"][1], 1),
                     beh["label"][0] / max(beh["label"][1], 1)])
        names.append(name)
        print(f"  transfer |rho| {transfer:.3f}  self R^2 {self_r2:.3f}")
    M = np.array(rows)
    heatmap(M, names, ["probe transfer", "self R^2", "order stab.", "label stab."],
            out_dir / "perturbation.png", f"E4 perturbations, layer {layer} pos {pos}", vmin=-0.2)
    np.savetxt(out_dir / "perturbation.csv", M, delimiter=",",
               header="variant rows: " + ",".join(names))
    manifest.add(layer=layer, pos=pos, variants=names, table=M.tolist())
    manifest.write()
    return out_dir


if __name__ == "__main__":
    args, cfg = parse_args(__doc__)
    run(cfg, args.out)
