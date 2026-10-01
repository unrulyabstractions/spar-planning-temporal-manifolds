"""Output locations, run manifests, and the activation file.

On the vast.ai box `WORKSPACE` is set and results go under
`$WORKSPACE/results/horizon/`. Locally they go under `out/` next to the
experiments.
"""

import hashlib
import json
import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

REGIONS = ("suffix", "think", "pre", "prelabel", "answer")


def out_base(out=None):
    """Base directory for run folders and the generation cache."""
    if out is not None:
        return Path(out)
    if os.environ.get("WORKSPACE"):
        return Path(os.environ["WORKSPACE"]) / "results" / "horizon"
    return Path(__file__).resolve().parent.parent / "out"


def resolve_out_dir(exp_name, cfg, out=None):
    """Run directory `<base>/<exp>_<model>_think-<mode>/`, created."""
    d = out_base(out) / f"{exp_name}_{cfg.tag}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=Path(__file__).parent, stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def code_hash():
    """Short sha256 over the package and experiment sources, so a manifest
    identifies the code even where there is no git checkout (the box)."""
    root = Path(__file__).resolve().parent.parent
    h = hashlib.sha256()
    for p in sorted(list((root / "spar_horizon").glob("*.py")) + list((root / "experiments").glob("*.py"))):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def stamp(msg):
    """Print `msg` with a wall-clock prefix so tmux logs show when phases end."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Manifest:
    """Collects run metadata; `write()` puts manifest.json in the run dir.
    `phase(name)` times a block and records it under `phases`."""

    def __init__(self, exp_name, cfg, out_dir):
        self.data = {"experiment": exp_name, "settings": cfg.to_dict(), "git": git_hash(),
                     "code_hash": code_hash(), "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                     "out_dir": str(out_dir), "phases": {}}
        self.out_dir = Path(out_dir)
        self._t0 = time.time()

    def add(self, **fields):
        self.data.update(fields)

    @contextmanager
    def phase(self, name):
        t0 = time.time()
        stamp(f"{name} ...")
        yield
        self.data["phases"][name] = round(time.time() - t0, 1)
        stamp(f"{name} done, {self.data['phases'][name]:.0f} s")

    def write(self):
        self.data["wall_seconds"] = round(time.time() - self._t0, 1)
        path = self.out_dir / "manifest.json"
        path.write_text(json.dumps(self.data, indent=2, default=_json_default))
        print(f"wrote {path}")
        return path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not serializable: {type(o)}")


def _opt(a, dtype=float):
    return np.array([], dtype=dtype) if a is None else np.asarray(a, dtype)


def save_activations(path, ext, horizons, rho=None, r2=None, far_delay_years=None, **extra):
    """Write the activation file: `L<i>` per layer, the kept layout (`tokens`,
    `n_suffix`, one `region_<name>` (start, end) pair per region,
    `think_fractions`), per-prompt metadata (`kept`, `horizons`, `forced_mask`,
    `think_len`, `lead`, `keep_idx`, `keep_ids`, `label_logits`), the sweeps
    (`rho`, `r2`), and any `extra` arrays (probe direction, tracks, label
    probe). Absent optional arrays are stored empty."""
    regions = {f"region_{k}": np.asarray(v, int) for k, v in ext.regions.items()}
    np.savez(path, tokens=np.array(ext.tokens), n_suffix=ext.n_suffix, kept=ext.kept,
             horizons=np.array([h if h is not None else np.nan for h in horizons]),
             think_fractions=np.asarray(ext.think_fractions, float),
             forced_mask=_opt(ext.forced_mask, bool), think_len=_opt(ext.think_len, int),
             lead=_opt(ext.lead, int), keep_idx=_opt(ext.keep_idx, int),
             keep_ids=_opt(ext.keep_ids, int), label_logits=_opt(ext.label_logits),
             rho=_opt(rho), r2=_opt(r2), far_delay_years=_opt(far_delay_years),
             **regions, **{k: _opt(v) for k, v in extra.items()},
             **{f"L{i}": a for i, a in enumerate(ext.activations)})
    print(f"wrote {path}")


def legacy_regions(n_suffix, n_positions):
    """Regions for an activation file written before think positions existed:
    everything before the answer is one turn block, the think and pre and
    pre-label regions are empty."""
    return {"suffix": (0, n_suffix), "think": (n_suffix, n_suffix), "pre": (n_suffix, n_suffix),
            "prelabel": (n_suffix, n_suffix), "answer": (n_suffix, n_positions)}


def load_activations(path):
    """Dict with `activations` (list per layer), `tokens`, `n_suffix`, `regions`,
    `think_fractions`, and every other array in the file under its own key.
    Files without regions get `legacy_regions`."""
    z = np.load(path, allow_pickle=False)
    layers = sorted(int(k[1:]) for k in z.files if k.startswith("L") and k[1:].isdigit())
    out = {"activations": [z[f"L{i}"] for i in layers], "tokens": [str(t) for t in z["tokens"]],
           "n_suffix": int(z["n_suffix"])}
    for k in z.files:
        if k in ("tokens", "n_suffix") or (k.startswith("L") and k[1:].isdigit()) or k.startswith("region_"):
            continue
        out[k] = z[k]
    if "region_suffix" in z.files:
        out["regions"] = {r: tuple(int(x) for x in z[f"region_{r}"]) for r in REGIONS}
    else:
        out["regions"] = legacy_regions(out["n_suffix"], len(out["tokens"]))
    if "think_fractions" not in z.files:
        out["think_fractions"] = np.array([])
    if "far_delay_years" not in z.files:
        out["far_delay_years"] = None
    return out
