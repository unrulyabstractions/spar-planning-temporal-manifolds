"""E0: the starter experiment on any model, with the horizon probe and 3D PCA.

    python experiments/exp0_baseline.py --model Qwen/Qwen3-8B --thinking off --limit 12

Writes to the run dir: geometry_2d.png, geometry_3d.png (and the horizon/far-
delay ratio versions), activations.npz (all layers, probe direction, tracks),
answers.json, manifest.json. With thinking on also probe_track.png (probe
reading at every kept position), dense_track.png (at every think token), and
label_probe.png (can the chosen label be read out before it is written?).
"""

import json

import numpy as np

from common import parse_args
from spar_horizon.behavior import behavior
from spar_horizon.extract import dense_track, extract
from spar_horizon.geometry import (display_layer, label_probe, label_probe_sweep, probe_directions,
                                   probe_sweep, probe_track, sweep)
from spar_horizon.model_io import load_model
from spar_horizon.plots import (dense_panel, label_probe_panel, pca3_panels, pca3_stats, pca_panels,
                                ratio_spec, track_panel)
from spar_horizon.prompts import DELAY_YEARS, build_prompts
from spar_horizon.runs import Manifest, resolve_out_dir, save_activations

EXP = "exp0"
MIN_NONFORCED = 8   # fewer non-forced prompts than this and the split sweeps are all nan
LABEL_POSITIONS = ("suffix probe pos", "think@1.0", "pre-label", "answer (ceiling)")


def label_positions(ext, probe_pos):
    """Kept-layout indices of the four positions the label probe compares."""
    return [probe_pos, ext.regions["think"][1] - 1, ext.regions["prelabel"][0], ext.regions["answer"][0]]


def _peak(curve):
    """(max, argmax) of a per-layer curve, (nan, None) when it is all nan."""
    if np.all(np.isnan(curve)):
        return float("nan"), None
    return float(np.nanmax(curve)), int(np.nanargmax(curve))


def think_report(ext, layer, rho, r2, rho_nf, r2_nf):
    """Per think fraction at `layer`: PC1 |rho| and probe R^2, all and non-forced prompts."""
    a = ext.regions["think"][0]
    return {f"{f:g}": {"rho": float(rho[layer, a + k]), "r2": float(r2[layer, a + k]),
                       "rho_nf": float(rho_nf[layer, a + k]), "r2_nf": float(r2_nf[layer, a + k])}
            for k, f in enumerate(ext.think_fractions)}


def run(cfg, out=None, skip_behavior=False):
    out_dir = resolve_out_dir(EXP, cfg, out)
    manifest = Manifest(EXP, cfg, out_dir)
    records = build_prompts(seed=cfg.seed, limit=cfg.limit)
    print(f"{len(records)} prompts, model {cfg.model}, thinking {cfg.thinking}")
    tokenizer, model, device = load_model(cfg)
    with manifest.phase("generate + extract"):
        ext = extract(tokenizer, model, device, [r["prompt"] for r in records], cfg)

    horizons = [r["horizon"] for r, k in zip(records, ext.kept) if k]
    has_horizon = np.array([h is not None for h in horizons])
    log_horizons = np.log10([h for h in horizons if h is not None])
    nf = ~ext.forced_mask

    with manifest.phase("pc1 sweep"):
        rho, Z_all = sweep(ext.activations, ext.tokens, has_horizon, log_horizons)
    peak = np.unravel_index(np.nanargmax(rho), rho.shape)
    layer = display_layer(rho)
    print(f"\nPEAK |rho| = {rho[peak]:.3f} at layer {peak[0]}, token {ext.tokens[peak[1]]!r}")
    print(f"display layer (best mean |rho| across tokens): {layer}")

    with manifest.phase("probe sweep"):
        r2 = probe_sweep(ext.activations, has_horizon, log_horizons)
    probe_layer, probe_pos = np.unravel_index(np.nanargmax(r2), r2.shape)
    print(f"probe: best CV R^2 = {r2[probe_layer, probe_pos]:.3f} at layer {probe_layer}, "
          f"token {ext.tokens[probe_pos]!r}")
    print("probe R^2 per layer (mean over positions): "
          + " ".join(f"{np.nanmean(r):.2f}" for r in r2))

    title = f"{cfg.model}, thinking {cfg.thinking}, layer {layer}"
    pca_panels(layer, rho, Z_all, ext.tokens, has_horizon, log_horizons, ext.regions,
               out_dir / "geometry_2d.png", title + ": PC1 vs horizon (gray = no horizon)")
    evr, rho3, Z3 = pca3_stats(ext.activations, layer, has_horizon, log_horizons)
    pca3_panels(layer, Z3, evr, rho3, ext.tokens, has_horizon, log_horizons, ext.regions,
                out_dir / "geometry_3d.png", title + ": PC1-3 (evr = explained variance)")
    far = [DELAY_YEARS[r["delays"][1]] for r, k in zip(records, ext.kept) if k]
    spec = ratio_spec(log_horizons, [f for f, h in zip(far, horizons) if h is not None])
    pca_panels(layer, rho, Z_all, ext.tokens, has_horizon, log_horizons, ext.regions,
               out_dir / "geometry_2d_ratio.png",
               title + ": PC1 vs horizon/far-delay (blue = far pays too late)", spec=spec)
    pca3_panels(layer, Z3, evr, rho3, ext.tokens, has_horizon, log_horizons, ext.regions,
                out_dir / "geometry_3d_ratio.png", title + ": PC1-3, horizon/far-delay", spec=spec)

    with manifest.phase("probe direction + track"):
        coef, intercept = probe_directions(ext.activations, has_horizon, log_horizons, probe_pos)
        track = probe_track(ext.activations, coef, intercept)
    extra = {"probe_coef": coef, "probe_intercept": intercept, "probe_track": track}
    manifest.add(n_prompts=len(records), n_kept=int(ext.kept.sum()),
                 skipped=int((~ext.kept).sum()), n_suffix=ext.n_suffix, tokens=ext.tokens,
                 regions=ext.regions, think_fractions=ext.think_fractions,
                 peak_rho=float(rho[peak]), peak_layer=int(peak[0]), peak_token=ext.tokens[peak[1]],
                 display_layer=layer, probe_layer=int(probe_layer), probe_pos=int(probe_pos),
                 probe_r2=float(r2[probe_layer, probe_pos]),
                 pc3_evr_at_display=evr.tolist(), pc3_rho_at_display=rho3.tolist(),
                 think_len_median=float(np.median(ext.think_lens)), forced=ext.forced)

    if cfg.thinking == "on":
        acts_nf = [X[nf] for X in ext.activations]
        log_nf = np.log10([h for h, f in zip(horizons, nf) if h is not None and f])
        rho_nf, r2_nf = np.full_like(rho, np.nan), np.full_like(r2, np.nan)
        acc_nf = np.full((ext.n_layers, len(LABEL_POSITIONS)), np.nan)
        enough_nf = int(nf.sum()) >= MIN_NONFORCED
        if enough_nf:
            with manifest.phase("non-forced sweeps"):
                rho_nf, _ = sweep(acts_nf, ext.tokens, has_horizon[nf], log_nf, verbose=False)
                r2_nf = probe_sweep(acts_nf, has_horizon[nf], log_nf)
        else:
            print(f"only {int(nf.sum())} non-forced prompts; the non-forced split is left nan")
        extra.update(rho_nf=rho_nf, r2_nf=r2_nf)
        track_panel(track[probe_layer], ext.tokens, ext.regions, has_horizon, log_horizons,
                    ext.forced_mask, out_dir / "probe_track.png",
                    f"{cfg.model}: probe reading along the kept positions, layer {probe_layer}")
        with manifest.phase("dense track"):
            dense = dense_track(model, ext, probe_layer, coef[probe_layer], intercept[probe_layer])
        extra["dense_track"] = dense
        dense_panel(dense, ext.think_len, has_horizon, log_horizons, ext.forced_mask,
                    out_dir / "dense_track.png",
                    f"{cfg.model}: probe reading at every think token, layer {probe_layer}")
        y = np.array([a.strip().startswith("b") for a in ext.answers if a is not None], int)
        positions = label_positions(ext, probe_pos)
        with manifest.phase("label probe"):
            acc = label_probe_sweep(ext.activations, y, positions)
            if enough_nf:
                acc_nf = label_probe_sweep(acts_nf, y[nf], positions)
        baseline = label_probe(ext.activations[0][:, 0], y)[1]
        extra.update(label_acc=acc, label_acc_nf=acc_nf)
        label_probe_panel(acc, baseline, LABEL_POSITIONS, out_dir / "label_probe.png",
                          f"{cfg.model}: reading the chosen label from the residual stream")
        manifest.add(think_positions=think_report(ext, probe_layer, rho, r2, rho_nf, r2_nf),
                     label_probe={lab: {"peak_acc": _peak(acc[:, j])[0], "peak_layer": _peak(acc[:, j])[1],
                                        "peak_acc_nf": _peak(acc_nf[:, j])[0],
                                        "peak_layer_nf": _peak(acc_nf[:, j])[1]}
                                  for j, lab in enumerate(LABEL_POSITIONS)},
                     label_baseline=float(baseline), dense_layer=int(probe_layer),
                     n_forced_kept=int(ext.forced_mask.sum()))

    save_activations(out_dir / "activations.npz", ext, horizons, rho=rho, r2=r2, far_delay_years=far,
                     **extra)
    kept_rows = iter(zip(ext.forced_mask, ext.think_len, ext.think_text))
    rows = []
    for r, a, k in zip(records, ext.answers, ext.kept):
        row = {"horizon": r["horizon"], "prompt": r["prompt"], "answer": a}
        if k:
            forced, tl, tt = next(kept_rows)
            row.update(label=(a.strip()[:1] if a else None), forced=bool(forced), think_len=int(tl),
                       think_text=tt)
        rows.append(row)
    (out_dir / "answers.json").write_text(json.dumps(rows, indent=1))

    if not skip_behavior:
        with manifest.phase("behavior"):
            manifest.add(behavior=behavior(tokenizer, model, device, records, ext.answers, cfg))
    manifest.write()
    return out_dir


if __name__ == "__main__":
    args, cfg = parse_args(__doc__, lambda p: p.add_argument("--skip-behavior", action="store_true"))
    run(cfg, args.out, args.skip_behavior)
