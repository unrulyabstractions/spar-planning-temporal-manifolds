"""E1: nuisance and specificity (plan row R4).

At the display layer, per position: |rho| of PC1 with log horizon vs with the
near reward and with the delay pair. Then probe transfer from the base bank to
the swapped-order and relabeled banks. Reuses E0's activations when present.

    python experiments/exp1_nuisance.py --model Qwen/Qwen3-8B --thinking off
"""

import json

import numpy as np
from scipy.stats import spearmanr

from common import parse_args
from spar_horizon.extract import extract
from spar_horizon.geometry import display_layer, horizon_probe, sweep
from spar_horizon.model_io import load_model
from spar_horizon.plots import heatmap, pca_panels
from spar_horizon.prompts import build_prompts, choice_prompt
from spar_horizon.runs import Manifest, load_activations, resolve_out_dir

EXP = "exp1"


def load_or_extract_base(tokenizer, model, device, records, cfg, out):
    e0 = resolve_out_dir("exp0", cfg, out) / "activations.npz"
    if e0.exists():
        print(f"using E0 activations from {e0}")
        z = load_activations(e0)
        return z["activations"], z["tokens"], z["kept"], z["regions"], int(json.loads(
            (e0.parent / "manifest.json").read_text())["display_layer"])
    ext = extract(tokenizer, model, device, [r["prompt"] for r in records], cfg)
    return ext.activations, ext.tokens, ext.kept, ext.regions, None


def run(cfg, out=None):
    out_dir = resolve_out_dir(EXP, cfg, out)
    manifest = Manifest(EXP, cfg, out_dir)
    records = build_prompts(seed=cfg.seed, limit=cfg.limit)
    tokenizer, model, device = load_model(cfg)
    acts, tokens, kept, regions, layer = load_or_extract_base(tokenizer, model, device, records, cfg, out)
    recs = [r for r, k in zip(records, kept) if k]
    has_h = np.array([r["horizon"] is not None for r in recs])
    log_h = np.log10([r["horizon"] for r in recs if r["horizon"] is not None])
    log_reward = np.log10([r["rewards"][0] for r in recs])
    delay_flag = np.array([r["delays"][0] == "1 month" for r in recs], dtype=float)

    rho, Z_all = sweep(acts, tokens, has_h, log_h, verbose=False)
    if layer is None:
        layer = display_layer(rho)
    n_pos = len(tokens)
    nuis = np.full((3, n_pos), np.nan)
    for pos in range(n_pos):
        Z = Z_all[layer][pos]
        if Z is None:
            continue
        nuis[0, pos] = rho[layer, pos]
        nuis[1, pos] = abs(spearmanr(Z[:, 0], log_reward)[0])
        nuis[2, pos] = abs(spearmanr(Z[:, 0], delay_flag)[0]) if delay_flag.std() > 0 else np.nan
    heatmap(nuis, ["horizon", "near reward", "delay pair"], [repr(t) for t in tokens],
            out_dir / "pc1_specificity.png", f"|rho| of PC1 at layer {layer} with each factor")
    pca_panels(layer, rho, Z_all, tokens, np.ones(len(recs), bool), log_reward, regions,
               out_dir / "geometry_2d_by_reward.png",
               f"{cfg.model}, thinking {cfg.thinking}, layer {layer}: colored by near reward")

    # probe transfer to swapped and relabeled banks at the probe position
    probe_pos = int(np.nanargmax(rho[layer]))
    X_base = acts[layer][has_h, probe_pos]
    _, _, probe = horizon_probe(X_base, log_h)
    transfer = {}
    h_recs = [r for r in recs if r["horizon"] is not None]
    for name, kw in (("swapped", {"swap": True}), ("relabeled", {"labels": ("1", "2")})):
        prompts = [choice_prompt(r["rewards"], r["delays"], r["horizon"], **kw) for r in h_recs]
        ext = extract(tokenizer, model, device, prompts, cfg, layers=[layer])
        X = ext.activations[0][:, probe_pos]
        lh = log_h[ext.kept]
        transfer[name] = float(abs(spearmanr(probe.predict(X), lh)[0]))
        print(f"probe transfer base -> {name}: |rho| = {transfer[name]:.3f}")

    manifest.add(display_layer=layer, probe_pos=probe_pos, probe_token=tokens[probe_pos],
                 specificity={"horizon": nuis[0].tolist(), "near_reward": nuis[1].tolist(),
                              "delay_pair": nuis[2].tolist()},
                 transfer=transfer, n_prompts=len(recs))
    manifest.write()
    return out_dir


if __name__ == "__main__":
    args, cfg = parse_args(__doc__)
    run(cfg, args.out)
