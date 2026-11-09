"""Reading the saved folder of an E0 run (`exp0_<model>_think-<mode>_<bank>`): its answers, the
settings it was built with, the prompt records those settings rebuild, and the decision per prompt.

Shared by the stages: stage 1 builds on these files, stage 3 reads them, and neither should
import the other's scripts for it. Reads saved files only; runs no model.
"""

import dataclasses
import json
from pathlib import Path

import numpy as np

from .banks import build_records
from .config import Settings
from .full_bank import natural_horizon

MODEL = "Qwen3-8B"
NORMAL_LEAD = 3     # the lead token count of an answer with no markdown before its label


def run_folder(root, model, mode, bank):
    """The E0 run folder of `model` for `mode` ("on"/"off") on `bank`; a results folder holds several."""
    return Path(root) / f"exp0_{model.split('/')[-1]}_think-{mode}_{bank}"


def run_dir(root, model, mode, bank="full"):
    """`run_folder`, which must hold a finished run."""
    run = run_folder(root, model, mode, bank)
    if not (run / "manifest.json").exists():
        raise FileNotFoundError(f"no finished E0 run at {run}")
    return run


def read_answers(run):
    return json.loads((Path(run) / "answers.json").read_text())


def manifest_settings(run):
    """The Settings an E0 run was built with, from its manifest (fields since renamed are ignored)."""
    manifest = json.loads((Path(run) / "manifest.json").read_text())
    known = {f.name for f in dataclasses.fields(Settings)}
    return Settings(**{k: v for k, v in manifest["settings"].items() if k in known})


def run_settings(root, model, mode, bank="full"):
    """The Settings of the E0 run in `root` for `model`, `mode` and `bank`, cache under root."""
    cfg = manifest_settings(run_dir(root, model, mode, bank))
    cfg.cache_dir = str(Path(root) / "gen_cache")
    return cfg


def run_records(root, run_name):
    """The prompt records of the E0 run folder `run_name` under `root`, by sample_uid."""
    return {r["sample_uid"]: r for r in build_records(manifest_settings(Path(root) / run_name))}


def single_rendering(row):
    """Pair labels for a full-bank row: one natural rendering, its own pair."""
    h = natural_horizon(row["horizon"])
    return {"pair_id": row["sample_uid"], "duration_id": str(h), "rendered_unit": h.unit, "natural": True}


def load_run_activations(root, mode, bank="units", model=MODEL):
    """The run's activation file and per-kept-prompt labels, or None. `bank` is "units"
    or "fill" (paired banks) or "full", whose prompts are single natural renderings,
    each its own pair. Rows without a horizon (the full bank's null prompts) are
    dropped; labels["rows"] indexes the kept rows in the activation file."""
    run = Path(root) / f"exp0_{model}_think-{mode}_{bank}"
    if not (run / "activations.npz").exists():
        return None
    z = np.load(run / "activations.npz")
    rows = [r for r in read_answers(run) if "think_len" in r]   # kept prompts
    if len(rows) != len(z["horizons"]):
        raise ValueError(f"{run}: {len(rows)} kept answers but {len(z['horizons'])} activation rows")
    keep = np.array([i for i, r in enumerate(rows) if r["horizon"] is not None])
    rows = [rows[i] if "pair_id" in rows[i] else {**rows[i], **single_rendering(rows[i])} for i in keep]
    natural_unit = {r["pair_id"]: r["rendered_unit"] for r in rows if r["natural"]}
    ab = z["label_logits"][:, 0] - z["label_logits"][:, 1]
    labels = {"y": np.log10(z["horizons"][keep]),
              "durations": np.array([r["duration_id"] for r in rows]),
              "units": np.array([r["rendered_unit"] for r in rows]),
              "natural_units": np.array([natural_unit[r["pair_id"]] for r in rows]),
              "pairs": np.array([r["pair_id"] for r in rows]),
              "natural": np.array([r["natural"] for r in rows]),
              "sample_uids": np.array([r["sample_uid"] for r in rows]),
              "regime": np.array([r["regime"] for r in rows]),
              "forced": np.array([bool(r.get("forced", False)) for r in rows]),
              "near_log_odds": np.where([r["near_label"] == "a" for r in rows], ab[keep], -ab[keep]),
              "chose_near": np.array([r["label"] == r["near_label"] for r in rows]),
              "anchor": np.array([bool(r.get("anchor", False)) for r in rows]),
              "rows": keep}
    return z, labels


def saved_decision(run, z):
    """The decision per horizon prompt of a saved E0 run (rows as in cell_data):
    a-vs-b log-odds, near-vs-far log-odds, chose_near. Rows whose lead is not NORMAL_LEAD
    have NaN log-odds: runs saved before extract.label_pair read the spaced ' a'/' b' logits
    there, but after a markdown lead like `**b)**` the label is unspaced, so those logits are
    for the wrong token pair. Callers drop the NaN rows."""
    answers = read_answers(run)
    keep = ~np.isnan(z["horizons"])
    rows = [a for a, k in zip(answers, keep) if k]
    ab = z["label_logits"][keep, 0] - z["label_logits"][keep, 1]
    ab[z["lead"][keep] != NORMAL_LEAD] = np.nan
    near_is_a = np.array([r["near_label"] == "a" for r in rows])
    near = np.where(near_is_a, ab, -ab)
    chose_near = np.array([r["label"] == r["near_label"] for r in rows])
    return ab, near, chose_near
