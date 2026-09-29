"""Export PC1-3 coordinates for every (mode, layer, position) of the E0 runs,
for the interactive 3D explorer.

    python analysis/export_pca3.py results/horizon_v1_noprefill  -> <root>/analysis/pca3.json
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from spar_horizon.prompts import DELAY_YEARS, build_prompts, horizon_text  # noqa: E402
from spar_horizon.runs import load_activations  # noqa: E402


def export(root):
    root = Path(root)
    out = {"modes": {}}
    for mode in ("off", "on"):
        d = root / f"exp0_Qwen3-8B_think-{mode}"
        if not (d / "activations.npz").exists():
            continue
        z = load_activations(d / "activations.npz")
        manifest = json.loads((d / "manifest.json").read_text())
        answers = {}
        if (d / "answers.json").exists():
            answers = {a["prompt"]: a["answer"] for a in json.loads((d / "answers.json").read_text())}
        recs = [r for r, k in zip(build_prompts(), z["kept"]) if k]
        has_h = ~np.isnan(z["horizons"])
        log_h = np.where(has_h, np.log10(np.where(has_h, z["horizons"], 1.0)), np.nan)
        far = np.array([DELAY_YEARS[r["delays"][1]] for r in recs])
        points = [{"h": (horizon_text(r["horizon"]) if r["horizon"] is not None else "no horizon"),
                   "lh": (None if not has_h[i] else round(float(log_h[i]), 3)),
                   "ratio": (None if not has_h[i] else round(float(log_h[i] - np.log10(far[i])), 3)),
                   "near": f"${r['rewards'][0]:,} in {r['delays'][0]}",
                   "far": f"${r['rewards'][1]:,} in {r['delays'][1]}",
                   "ans": (answers.get(r["prompt"], "") or "").strip()[:24]}
                  for i, r in enumerate(recs)]
        layers = []
        for l, X in enumerate(z["activations"]):
            per_pos = []
            for p in range(X.shape[1]):
                Xp = X[:, p] - X[:, p].mean(0)
                if np.allclose(Xp, 0):
                    per_pos.append(None)
                    continue
                pca = PCA(n_components=3).fit(Xp)
                Z = pca.transform(Xp)
                Z = Z / (np.abs(Z).max() or 1)
                rho = [abs(float(spearmanr(Z[has_h, k], log_h[has_h])[0])) for k in range(3)]
                per_pos.append({"xyz": np.round(Z, 3).tolist(),
                                "evr": np.round(pca.explained_variance_ratio_, 3).tolist(),
                                "rho": [round(r, 3) for r in rho]})
            layers.append(per_pos)
            print(f"{mode} L{l} done")
        out["modes"][mode] = {"tokens": z["tokens"], "n_suffix": z["n_suffix"], "points": points,
                              "layers": layers,
                              "meta": {k: manifest.get(k) for k in
                                       ("probe_layer", "probe_pos", "probe_r2", "peak_layer", "peak_rho",
                                        "display_layer", "forced", "think_len_median")}}
    path = root / "analysis" / "pca3.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return path


if __name__ == "__main__":
    export(sys.argv[1] if len(sys.argv) > 1 else "results/horizon")
