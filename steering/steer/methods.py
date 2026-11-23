"""Steering methods as self-describing bundles: the hand-over format to the evaluation team.

    methods/<model>/<method_id>/
        method.json      what the method is and how to apply it: kind, edit, layer, site, default positions, the
                         request x (name, unit, sign), what it was fitted on (model, bank, activations, settings,
                         hashes), how it was trained, the code commits, and the SHA-256 of every other file
        arrays.npz       the method's parameters: a unit direction and typical norm, or the curve (mu, U, t_grid, ...)
        fit.json         the fit's diagnostics and settings (copied from fit.py's output)
        fit_uids.jsonl   the prompts it was fitted on, with their classes (copied)
        search.jsonl     every fit.py run on the same activations up to the bundling (copied snapshot)

    method = load("methods/Qwen3.5-2B/caa__qwen35-2b-off__default__L15_label")
    method.edit                      # a steer.edits edit: method.edit.delta(h, x, ctx)
    with steering(model, method.edit, x=0.1, layer=method.layer, positions=method.positions): ...
    method.refit(activations_dir, bank_dir, uids=..., horizons=...)   # the same recipe on other prompts or labels

Methods bundled from one fit (bundle.py): `caa` (the horizon direction), `curve-snap` and `curve-displace` (the
manifold slide, two readings), and two concept controls fitted by the same recipe: `reward` (reward contrast) and
`choice` (baseline chose far minus chose near).
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from pbank.durations import Duration
from pbank.records import load_prompts, prompts_sha256, read_jsonl
from steer import activations
from steer.activations import file_sha256
from steer.edits import CurveEdit, LinearEdit
from steer.fitting import FIT_REGIMES, WITHIN_KEY, FitResult, FitSettings, fit, within_key
from steer.manifold import Manifold

FORMAT = "steer-method/1"
LINEAR = {   # method -> (direction key in directions.npz, what +x moves toward, how it is computed)
    "caa": ("caa", "longer horizons (mean of H/D >= caa_long_min minus mean of H/D <= caa_short_max)",
            "difference of class means, unit-normalized"),
    "reward": ("reward", "larger reward multipliers (mean of multiplier >= reward_high_min minus <= reward_low_max)",
               "difference of class means, unit-normalized (concept control)"),
    "choice": ("choice", "the far option as the baseline's choice (mean of chose far minus chose near)",
               "difference of class means, unit-normalized (concept control)"),
}
for _regime in FIT_REGIMES:   # the within-regime CAAs (comparison methods; fit.py's docstring defines them)
    LINEAR[f"caa-{_regime}"] = (
        within_key(_regime), f"longer horizons within the `{_regime}` regime (mean of the regime's prompts with the "
        "longest horizons minus mean of those with the shortest; each class is the within_quantile share of the regime)",
        "difference of class means within one regime, unit-normalized")
LINEAR["caa-within"] = (
    WITHIN_KEY, "longer horizons within a regime (the within-regime mean differences of the within_pool regimes, "
    "averaged with regime-size weights)", "regime-size-weighted average of within-regime differences of class means, "
    "unit-normalized")
CURVE = {"curve-snap": "snap", "curve-displace": "displace"}
METHODS = tuple(LINEAR) + tuple(CURVE)


def available_methods(fit_dir: Path) -> list[str]:
    """The methods a fit.py output can be bundled into: every curve method, and each linear one whose direction the
    fit has (a within-regime CAA is absent when its regime was left out or had too few distinct horizons)."""
    with np.load(Path(fit_dir) / "directions.npz") as z:
        have = set(z.files)
    return [m for m in METHODS if m in CURVE or LINEAR[m][0] in have]
LAYER_CONVENTION = ("residual stream after decoder block `layer`, counting blocks from 1 (= hidden_states[layer]); "
                    "see steering/steer/layers.py")


@dataclass
class Method:
    meta: dict                       # method.json
    edit: object                     # LinearEdit or CurveEdit
    structure: dict                  # the parameters: {"direction", "typical_norm"} or {"manifold": Manifold}
    path: Optional[Path] = None      # the bundle directory (None for an in-memory refit)

    @property
    def kind(self) -> str:
        return self.meta["kind"]

    @property
    def layer(self) -> int:
        return self.meta["layer"]

    @property
    def site(self) -> str:
        return self.meta["site"]

    @property
    def positions(self) -> str:
        return self.meta["default_positions"]

    def fit_uids(self) -> list[dict]:
        return read_jsonl(self.path / "fit_uids.jsonl") if self.path else self.meta.get("fit_uids", [])

    def refit(self, activations_dir: Path, bank_dir: Path, uids: Optional[list[str]] = None,
              horizons: Optional[dict] = None) -> "Method":
        """The same method, refitted with the same settings on other prompts and/or other horizon labels.

        uids: the prompts to fit on (default: the ones this method was fitted on). horizons: {uid: grid value},
        replacing those prompts' stated horizons (and so their H/D and log-time); e.g. a within-stratum shuffle for
        a label-shuffled null. Regimes are not recomputed: they only enter diagnostics. The activations must be the
        ones the method was fitted from (checked by hash). As in fit.py, verification-grid and no-horizon prompts are
        refused."""
        acts = activations.load(activations_dir, self.layer, self.site)
        fitted_on = self.meta["fitted_on"]
        array = activations.array_name(self.layer, self.site)
        if acts.meta["arrays"][array]["sha256"] != fitted_on["activations_array_sha256"]:
            raise ValueError(f"{activations_dir} is not the activation set {self.meta['method_id']} was fitted from")
        records = {r.uid: r for r in load_prompts(bank_dir)}
        if prompts_sha256(records.values()) != fitted_on["bank_prompts_sha256"]:
            raise ValueError(f"{bank_dir} is not the bank {self.meta['method_id']} was fitted on")
        uids = list(uids) if uids is not None else [row["uid"] for row in self.fit_uids()]
        row_of = {uid: i for i, uid in enumerate(acts.uids)}
        missing = [u for u in uids if u not in row_of]
        if missing:
            raise ValueError(f"{len(missing)} uids have no activations, e.g. {missing[:3]}")
        unusable = [u for u in uids if records[u].split != "extraction" or records[u].horizon is None]
        if unusable:
            raise ValueError(f"{len(unusable)} uids are verification-grid or no-horizon prompts, which are never "
                             f"fitted on, e.g. {unusable[:3]}")
        recs = [relabel(records[u], (horizons or {}).get(u)) for u in uids]
        rows = [row_of[u] for u in uids]
        chosen = np.array([acts.index[i]["chosen_option"] for i in rows])
        result = fit(acts.X[rows], recs, chosen, FitSettings(**self.meta["fit_settings"]))
        meta = {**self.meta, "method_id": self.meta["method_id"] + "__refit",
                "refit_of": self.meta["method_id"], "refit": {"n_prompts": len(uids),
                                                              "relabelled": len(horizons or {})},
                "fit_uids": [{"uid": u, "horizon_grid_value": r.horizon_grid_value} for u, r in zip(uids, recs)]}
        return from_fit_result(meta, result)


def relabel(record, horizon: Optional[str]):
    """The record with its stated horizon replaced by grid value `horizon` (H/D as the bank computes it:
    horizon days / far delay days). None returns the record unchanged."""
    if horizon is None:
        return record
    days = Duration.parse(horizon).days
    return dataclasses.replace(record, horizon=horizon, horizon_grid_value=horizon,
                               h_over_far_delay=float(days / Duration.parse(record.far.delay).days))


def from_fit_result(meta: dict, result: FitResult) -> Method:
    """The in-memory method of `meta["method"]` from a FitResult (used by refit)."""
    name = meta["method"]
    if name in LINEAR:
        direction = result.direction(LINEAR[name][0]).vector
        structure = {"direction": direction, "typical_norm": result.typical_norm}
        return Method(meta=meta, edit=LinearEdit(direction, result.typical_norm), structure=structure)
    return Method(meta=meta, edit=CurveEdit(result.manifold, CURVE[name]), structure={"manifold": result.manifold})


def load(path: Path) -> Method:
    """A bundle written by bundle.py, after checking every file against the SHA-256 recorded in method.json."""
    path = Path(path)
    meta = json.loads((path / "method.json").read_text())
    if meta.get("format") != FORMAT:
        raise ValueError(f"{path}: not a {FORMAT} bundle")
    for name, digest in meta["files"].items():
        if file_sha256(path / name) != digest:
            raise ValueError(f"{path / name} does not match the SHA-256 recorded in method.json")
    with np.load(path / "arrays.npz") as z:
        arrays = {k: z[k] for k in z.files}
    if meta["method"] in LINEAR:
        structure = {"direction": arrays["direction"], "typical_norm": float(arrays["typical_norm"])}
        edit = LinearEdit(structure["direction"], structure["typical_norm"])
    else:
        manifold = Manifold.load(path / "arrays.npz")
        structure = {"manifold": manifold}
        edit = CurveEdit(manifold, CURVE[meta["method"]])
    return Method(meta=meta, edit=edit, structure=structure, path=path)


def write_bundle(out: Path, method: str, fit_dir: Path, fit_meta: dict, extract_meta: dict, search_log: Path,
                 code_commit: str, created_utc: str) -> Path:
    """Write one method's bundle from a fit.py output directory (see bundle.py)."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    out.mkdir(parents=True)
    s = fit_meta["settings"]
    if method in LINEAR:
        key, toward, training = LINEAR[method]
        with np.load(fit_dir / "directions.npz") as z:
            if key not in z.files:
                raise ValueError(f"{fit_dir} has no {key} direction, so {method} cannot be bundled")
            direction, norm = z[key], float(z["typical_norm"])
        np.savez(out / "arrays.npz", direction=direction, typical_norm=norm)
        kind, edit = "push", {"class": "LinearEdit"}
        request = {"name": "strength", "unit": "fraction of typical_norm (the mean activation norm at this layer and site)",
                   "sign": f"+x moves toward {toward}", "typical_norm": norm}
    else:
        shutil.copyfile(fit_dir / "manifold.npz", out / "arrays.npz")
        m = Manifold.load(out / "arrays.npz")
        kind, edit, training = "target", {"class": "CurveEdit", "mode": CURVE[method]}, (
            "closed form: PCA of the per-horizon means, Gaussian-kernel local-linear curve in log10(days)")
        request = {"name": "shift", "unit": "decades of log10(horizon in days), from each token's t-hat",
                   "sign": "+x moves toward longer horizons", "t_range_log10_days": list(m.t_range),
                   "clamped": "t-hat + x is clamped to t_range"}
    shutil.copyfile(fit_dir / "fit.json", out / "fit.json")
    shutil.copyfile(fit_dir / "fit_uids.jsonl", out / "fit_uids.jsonl")
    shutil.copyfile(search_log, out / "search.jsonl")
    array = fit_meta["inputs"]["array"]
    meta = {
        "format": FORMAT, "method_id": out.name, "method": method, "kind": kind, "edit": edit,
        "layer": s["layer"], "layer_convention": LAYER_CONVENTION, "site": s["site"],
        "default_positions": "all", "positions_note": "the report steers all tokens and checks generated tokens only; "
                                                      "steer.hooks.steering takes all / prompt / generated / a callable",
        "request": request,
        "training": {"procedure": training, "objective": None, "selected_by_search": False,
                     "search_log": "search.jsonl lists every fit.py run on these activations up to bundling"},
        "fitted_on": {"model": extract_meta["model"], "thinking": extract_meta["run_settings"]["thinking"],
                      "decoding": extract_meta["run_settings"].get("decoding", "greedy"),
                      "baseline_run": extract_meta["run_name"], "bank": extract_meta["bank_name"],
                      "bank_prompts_sha256": extract_meta["settings"]["bank_prompts_sha256"],
                      "activations": fit_meta["inputs"]["activations"], "activations_array": array,
                      "activations_array_sha256": fit_meta["inputs"]["array_sha256"],
                      "fit_name": s["name"], "n_prompts": fit_meta["n_prompts"], "regimes": fit_meta["regimes"],
                      "excluded": fit_meta["excluded"], "include_none": s["include_none"]},
        # A fit written before a setting existed lacks it; the setting's default is then what that fit used.
        "fit_settings": {f.name: s.get(f.name, f.default) for f in dataclasses.fields(FitSettings)},
        "code_commit": {"extract": extract_meta["code_commit"], "fit": fit_meta["code_commit"], "bundle": code_commit},
        "created_utc": created_utc,
    }
    meta["files"] = {name: file_sha256(out / name)
                     for name in ("arrays.npz", "fit.json", "fit_uids.jsonl", "search.jsonl")}
    (out / "method.json").write_text(json.dumps(meta, indent=2) + "\n")
    return out
