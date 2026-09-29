"""E3: unit invariance (plan row R3).

Six durations, three spellings each ("1 year" / "12 months" / "365 days").
Project onto the horizon probe trained on E0's bank and compare the spread
across spellings of one duration with the spread across durations. If the axis
encodes log-time, spellings of the same duration land together.

    python experiments/exp3_units.py --model Qwen/Qwen3-8B --thinking off
"""

import json

import matplotlib.pyplot as plt
import numpy as np

from common import parse_args
from spar_horizon.extract import extract
from spar_horizon.geometry import horizon_probe
from spar_horizon.model_io import load_model
from spar_horizon.prompts import UNIT_SPELLINGS, build_unit_prompts, horizon_text
from spar_horizon.runs import Manifest, load_activations, resolve_out_dir

EXP = "exp3"


def probe_from_e0(cfg, out):
    d = resolve_out_dir("exp0", cfg, out)
    if not (d / "activations.npz").exists():
        return None
    m = json.loads((d / "manifest.json").read_text())
    z = load_activations(d / "activations.npz")
    layer, pos = m["probe_layer"], m["probe_pos"]
    has_h = ~np.isnan(z["horizons"])
    _, _, probe = horizon_probe(z["activations"][layer][has_h, pos], np.log10(z["horizons"][has_h]))
    print(f"probe from E0: layer {layer}, position {pos}")
    return layer, pos, probe


def run(cfg, out=None):
    out_dir = resolve_out_dir(EXP, cfg, out)
    manifest = Manifest(EXP, cfg, out_dir)
    tokenizer, model, device = load_model(cfg)
    records = build_unit_prompts()
    if cfg.limit:  # keep whole spelling groups: one reward/delay combo, first durations
        first = (records[0]["rewards"], records[0]["delays"])
        records = [r for r in records if (r["rewards"], r["delays"]) == first][:cfg.limit]
    e0 = probe_from_e0(cfg, out)
    layers = [e0[0]] if e0 else None
    ext = extract(tokenizer, model, device, [r["prompt"] for r in records], cfg, layers=layers)
    recs = [r for r, k in zip(records, ext.kept) if k]
    log_h = np.log10([r["horizon"] for r in recs])
    if e0:
        layer, pos, probe = e0
        X = ext.activations[0][:, min(pos, ext.activations[0].shape[1] - 1)]
        source = "E0"
    else:  # self-fit fallback: best layer at the last suffix position
        pos = ext.n_suffix - 1
        r2s = [horizon_probe(a[:, pos], log_h)[0] for a in ext.activations]
        layer = int(np.nanargmax(r2s))
        X = ext.activations[layer][:, pos]
        _, _, probe = horizon_probe(X, log_h)
        source = "self"
    proj = probe.predict(X)

    # spreads along the probe axis
    by_dur = {}
    for r, p in zip(recs, proj):
        by_dur.setdefault(r["horizon"], {}).setdefault(r["spelling"], []).append(p)
    spelling_means = {d: {s: float(np.mean(v)) for s, v in sp.items()} for d, sp in by_dur.items()}
    within = float(np.mean([np.std(list(m.values())) for m in spelling_means.values() if len(m) > 1]))
    dur_means = [np.mean(list(m.values())) for m in spelling_means.values()]
    between = float(np.std(dur_means)) if len(dur_means) > 1 else float("nan")
    ratio = within / between if between else float("nan")
    print(f"within-duration spread {within:.3f}  between-duration spread {between:.3f}  "
          f"ratio {ratio:.3f}  (pass < 0.2)")

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (d, m) in enumerate(sorted(spelling_means.items())):
        for s, v in m.items():
            ax.scatter(i, v, s=30)
            ax.annotate(s, (i, v), fontsize=7, xytext=(4, 0), textcoords="offset points")
    ax.set_xticks(range(len(spelling_means)), [horizon_text(d) for d in sorted(spelling_means)])
    ax.set_ylabel("probe prediction (log10 years)")
    ax.set_title(f"unit spellings on the horizon axis, layer {layer} ({source} probe)  "
                 f"within/between = {ratio:.2f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "units.png", dpi=150)
    plt.close(fig)
    manifest.add(layer=layer, pos=pos, probe_source=source, n_prompts=len(recs),
                 within=within, between=between, ratio=ratio,
                 spelling_means={horizon_text(d): m for d, m in spelling_means.items()})
    manifest.write()
    return out_dir


if __name__ == "__main__":
    args, cfg = parse_args(__doc__)
    run(cfg, args.out)
