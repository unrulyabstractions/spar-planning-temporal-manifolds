"""Output locations and run manifests.

On the vast.ai box `WORKSPACE` is set and results go under
`$WORKSPACE/results/horizon/`, which `just sync` pulls home. Locally they go
under `out/` next to the experiments.
"""

import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np


def resolve_out_dir(exp_name, cfg, out=None):
    if out is not None:
        base = Path(out)
    elif os.environ.get("WORKSPACE"):
        base = Path(os.environ["WORKSPACE"]) / "results" / "horizon"
    else:
        base = Path(__file__).resolve().parent.parent / "out"
    d = base / f"{exp_name}_{cfg.tag}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=Path(__file__).parent, stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


class Manifest:
    """Collects run metadata; `write()` puts manifest.json in the run dir."""

    def __init__(self, exp_name, cfg, out_dir):
        self.data = {"experiment": exp_name, "settings": cfg.to_dict(), "git": git_hash(),
                     "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "out_dir": str(out_dir)}
        self.out_dir = Path(out_dir)
        self._t0 = time.time()

    def add(self, **fields):
        self.data.update(fields)

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


def save_activations(path, ext, horizons, rho=None, r2=None, far_delay_years=None):
    np.savez(path, tokens=np.array(ext.tokens), n_suffix=ext.n_suffix, kept=ext.kept,
             horizons=np.array([h if h is not None else np.nan for h in horizons]),
             rho=rho if rho is not None else np.array([]),
             r2=r2 if r2 is not None else np.array([]),
             far_delay_years=np.array([]) if far_delay_years is None else np.asarray(far_delay_years, float),
             **{f"L{i}": a for i, a in enumerate(ext.activations)})
    print(f"wrote {path}")


def load_activations(path):
    z = np.load(path, allow_pickle=False)
    layers = sorted(int(k[1:]) for k in z.files if k.startswith("L"))
    return {"activations": [z[f"L{i}"] for i in layers], "tokens": [str(t) for t in z["tokens"]],
            "n_suffix": int(z["n_suffix"]), "horizons": z["horizons"], "kept": z["kept"],
            "rho": z["rho"], "r2": z["r2"],
            "far_delay_years": z["far_delay_years"] if "far_delay_years" in z.files else None}
