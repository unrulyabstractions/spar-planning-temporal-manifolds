"""E0: the starter experiment on any model, with the horizon probe and 3D PCA.

    python experiments/exp0_baseline.py --model Qwen/Qwen3-8B --thinking off --limit 12

Writes to the run dir: geometry_2d.png, geometry_3d.png, activations.npz
(all layers), probe direction at the best layer, manifest.json.
"""

import json

import numpy as np

from common import parse_args
from spar_horizon.behavior import behavior
from spar_horizon.extract import extract
from spar_horizon.geometry import display_layer, probe_sweep, sweep
from spar_horizon.model_io import load_model
from spar_horizon.plots import pca3_panels, pca3_stats, pca_panels, ratio_spec
from spar_horizon.prompts import DELAY_YEARS, build_prompts
from spar_horizon.runs import Manifest, resolve_out_dir, save_activations

EXP = "exp0"


def run(cfg, out=None, skip_behavior=False):
    out_dir = resolve_out_dir(EXP, cfg, out)
    manifest = Manifest(EXP, cfg, out_dir)
    records = build_prompts(seed=cfg.seed, limit=cfg.limit)
    print(f"{len(records)} prompts, model {cfg.model}, thinking {cfg.thinking}")
    tokenizer, model, device = load_model(cfg)
    ext = extract(tokenizer, model, device, [r["prompt"] for r in records], cfg)

    horizons = [r["horizon"] for r, k in zip(records, ext.kept) if k]
    has_horizon = np.array([h is not None for h in horizons])
    log_horizons = np.log10([h for h in horizons if h is not None])

    rho, Z_all = sweep(ext.activations, ext.tokens, has_horizon, log_horizons)
    peak = np.unravel_index(np.nanargmax(rho), rho.shape)
    layer = display_layer(rho)
    print(f"\nPEAK |rho| = {rho[peak]:.3f} at layer {peak[0]}, token {ext.tokens[peak[1]]!r}")
    print(f"display layer (best mean |rho| across tokens): {layer}")

    r2 = probe_sweep(ext.activations, has_horizon, log_horizons)
    probe_layer, probe_pos = np.unravel_index(np.nanargmax(r2), r2.shape)
    print(f"probe: best CV R^2 = {r2[probe_layer, probe_pos]:.3f} at layer {probe_layer}, "
          f"token {ext.tokens[probe_pos]!r}")
    print("probe R^2 per layer (mean over positions): "
          + " ".join(f"{np.nanmean(r):.2f}" for r in r2))

    title = f"{cfg.model}, thinking {cfg.thinking}, layer {layer}"
    pca_panels(layer, rho, Z_all, ext.tokens, has_horizon, log_horizons, ext.n_suffix,
               out_dir / "geometry_2d.png", title + ": PC1 vs horizon (gray = no horizon)")
    evr, rho3, Z3 = pca3_stats(ext.activations, layer, has_horizon, log_horizons)
    pca3_panels(layer, Z3, evr, rho3, ext.tokens, has_horizon, log_horizons, ext.n_suffix,
                out_dir / "geometry_3d.png", title + ": PC1-3 (evr = explained variance)")
    far = [DELAY_YEARS[r["delays"][1]] for r, k in zip(records, ext.kept) if k]
    spec = ratio_spec(log_horizons, [f for f, h in zip(far, horizons) if h is not None])
    pca_panels(layer, rho, Z_all, ext.tokens, has_horizon, log_horizons, ext.n_suffix,
               out_dir / "geometry_2d_ratio.png",
               title + ": PC1 vs horizon/far-delay (blue = far pays too late)", spec=spec)
    pca3_panels(layer, Z3, evr, rho3, ext.tokens, has_horizon, log_horizons, ext.n_suffix,
                out_dir / "geometry_3d_ratio.png", title + ": PC1-3, horizon/far-delay", spec=spec)
    save_activations(out_dir / "activations.npz", ext, horizons, rho=rho, r2=r2, far_delay_years=far)
    (out_dir / "answers.json").write_text(json.dumps(
        [{"horizon": r["horizon"], "prompt": r["prompt"], "answer": a}
         for r, a in zip(records, ext.answers)], indent=1))

    manifest.add(n_prompts=len(records), n_kept=int(ext.kept.sum()),
                 skipped=int((~ext.kept).sum()), n_suffix=ext.n_suffix, tokens=ext.tokens,
                 peak_rho=float(rho[peak]), peak_layer=int(peak[0]), peak_token=ext.tokens[peak[1]],
                 display_layer=layer, probe_layer=int(probe_layer), probe_pos=int(probe_pos),
                 probe_r2=float(r2[probe_layer, probe_pos]),
                 pc3_evr_at_display=evr.tolist(), pc3_rho_at_display=rho3.tolist(),
                 think_len_median=float(np.median(ext.think_lens)), forced=ext.forced)
    if not skip_behavior:
        manifest.add(behavior=behavior(tokenizer, model, device, records, ext.answers, cfg))
    manifest.write()
    return out_dir


if __name__ == "__main__":
    args, cfg = parse_args(__doc__, lambda p: p.add_argument("--skip-behavior", action="store_true"))
    run(cfg, args.out, args.skip_behavior)
