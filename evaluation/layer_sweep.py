"""Layer x site table of how well the horizon can be read and a CAA direction fitted (CPU only, offline).

For every L<layer>_<site>.npy in an activations directory, on the extraction prompts that state a horizon (regimes
near_only and both, no `none`; with --uids, only the prompts of that file's role, e.g. the reserved-horizon fit set):
  ridge rho    held-out-horizon Spearman of a ridge probe of log10 horizon on the activations (curve_dim.py's folds);
  curve rho    the same for the steering curve with K components;
  reliability  median cosine between CAA directions refitted on two random halves of the prompts (N_SPLITS halvings,
               diagnose.fit_direction), CAA = mean(H/D >= 4) - mean(H/D <= 0.25) as fit.py defines it;
  class gap    far-choice rate of the long class minus that of the short class (the baseline's own choices).
Writes layer_sweep.json and layer_sweep.html (one table per metric, cells shaded by value; light and dark theme).

Usage:
  python layer_sweep.py <activations dir> --bank prompt_bank/data/t0_s0 [--uids splits/reserved_horizons.jsonl
      [--uids-role fit]] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm

import curve_dim
import diagnose

CURVE_K = 4
CAA_LONG_MIN, CAA_SHORT_MAX = 4.0, 0.25      # fit.py's FitSettings
SCANNED_REGIMES = ("near_only", "both")
METRICS = {"ridge_rho": "ridge probe: held-out-horizon rho", "curve_rho": f"curve (k = {CURVE_K}): held-out-horizon rho",
           "caa_reliability": f"CAA split-half reliability (median cosine of {diagnose.N_SPLITS} halvings)",
           "class_gap": "CAA classes: far-rate gap (long minus short)"}


def select_rows(index_path, prompts_path, uids_path=None, role="fit"):
    """Prompts to scan, each with its activation row and baseline choice."""
    prompts = {r["uid"]: r for r in diagnose.read_jsonl(prompts_path)}
    allowed = None if uids_path is None else {r["uid"] for r in diagnose.read_jsonl(uids_path) if r["role"] == role}
    rows = []
    for i, entry in enumerate(diagnose.read_jsonl(index_path)):
        p = prompts.get(entry["uid"])
        if p and p["split"] == "extraction" and p["horizon"] and p["regime"] in SCANNED_REGIMES and (allowed is None or p["uid"] in allowed):
            rows.append({**p, "row": entry.get("row", i), "chosen_option": entry.get("chosen_option")})
    return rows


def caa_numbers(X, rows):
    """(split-half reliability, far-rate gap) of the CAA direction on activations X of `rows`."""
    hd = np.array([r["h_over_far_delay"] for r in rows])
    sign = np.where(hd >= CAA_LONG_MIN, 1, np.where(hd <= CAA_SHORT_MAX, -1, 0))
    group = np.zeros(len(rows), int)
    rng = np.random.default_rng(0)
    halves = []
    for _ in range(diagnose.N_SPLITS):
        perm = rng.permutation(len(rows))
        a, b = (diagnose.fit_direction(X, sign, group, perm[:len(rows) // 2]), diagnose.fit_direction(X, sign, group, perm[len(rows) // 2:]))
        halves.append(diagnose.cosine(a, b))
    halves = [h for h in halves if h is not None]
    far = np.array([r["chosen_option"] == "far" for r in rows])
    parsed = np.array([r["chosen_option"] in ("far", "near") for r in rows])
    rate = lambda mask: far[mask & parsed].mean() if (mask & parsed).any() else np.nan
    return (float(np.median(halves)) if halves else None), float(rate(sign == 1) - rate(sign == -1))


def scan_file(path, rows):
    X = np.load(path, mmap_mode="r")[[r["row"] for r in rows]].astype(np.float64)
    groups = [r["horizon_grid_value"] for r in rows]
    t = np.array([curve_dim.log10_days(g) for g in groups])
    folds = curve_dim.horizon_folds(groups)
    ridge_hat = np.empty(len(X))
    for test in folds:
        ridge_hat[test] = curve_dim.ridge_predict(X[~test], t[~test], X[test])
    curve_hat = curve_dim.cross_validated_t(X, t, groups, folds, CURVE_K, curve_dim.BANDWIDTH)
    reliability, gap = caa_numbers(X, rows)
    return {"ridge_rho": curve_dim.spearman(t, ridge_hat), "curve_rho": curve_dim.spearman(t, curve_hat),
            "caa_reliability": reliability, "class_gap": gap}


def heat_table(metric, cells, layers, sites):
    def td(layer, site):
        v = cells.get((layer, site), {}).get(metric)
        if v is None or np.isnan(v):
            return "<td>-</td>"
        return f'<td class="heat" style="--v:{55 * min(max(v, 0), 1):.0f}%">{v:.2f}</td>'
    head = "".join(f"<th>{diagnose.esc(s)}</th>" for s in sites)
    body = "".join(f"<tr><th>L{layer}</th>{''.join(td(layer, s) for s in sites)}</tr>" for layer in layers)
    return f'<h2>{diagnose.esc(METRICS[metric])}</h2><div class="scroll"><table><tr><th></th>{head}</tr>{body}</table></div>'


def render(cells, layers, sites, n_prompts, source):
    css = diagnose.CSS + "td.heat{background:color-mix(in srgb,var(--accent) var(--v),var(--bg))}"
    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            f"<title>Layer sweep</title><style>{css}</style></head><body><main><h1>Layer sweep</h1>"
            f"<p class=\"muted\">{diagnose.esc(source)}: {n_prompts} prompts with a horizon (no `none`). Held-out-horizon folds hold out "
            "whole horizons. Shading is the value from 0 to 1 (negative values unshaded). Reports, never gates.</p>"
            + "".join(heat_table(m, cells, layers, sites) for m in METRICS) + "</main></body></html>")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("activations")
    ap.add_argument("--bank", required=True, help="prompt bank directory with prompts.jsonl")
    ap.add_argument("--uids", help="a splits/*.jsonl file with uid and role")
    ap.add_argument("--uids-role", default="fit")
    ap.add_argument("--out", help="default: the activations directory")
    a = ap.parse_args(argv)
    acts = Path(a.activations)
    out = Path(a.out) if a.out else acts
    out.mkdir(parents=True, exist_ok=True)
    rows = select_rows(acts / "index.jsonl", Path(a.bank) / "prompts.jsonl", Path(a.uids) if a.uids else None, a.uids_role)
    files = sorted((int(m[1]), m[2], p) for p in acts.glob("L*_*.npy") if (m := re.fullmatch(r"L(\d+)_(.+)\.npy", p.name)))
    cells = {(layer, site): scan_file(p, rows) for layer, site, p in tqdm(files, desc="layer sweep", unit="file")}
    layers, sites = sorted({k[0] for k in cells}), sorted({k[1] for k in cells})
    source = f"{acts}" + (f", {a.uids_role} role of {Path(a.uids).name}" if a.uids else "")
    (out / "layer_sweep.json").write_text(json.dumps(
        {"source": source, "n_prompts": len(rows), "curve_k": CURVE_K,
         "cells": [{"layer": layer, "site": site, **v} for (layer, site), v in cells.items()]}, indent=2) + "\n")
    (out / "layer_sweep.html").write_text(render(cells, layers, sites, len(rows), source))
    print(f"wrote {out / 'layer_sweep.json'} and layer_sweep.html ({len(files)} files, {len(rows)} prompts)")
    return cells


if __name__ == "__main__":
    main()
