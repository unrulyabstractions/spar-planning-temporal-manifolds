"""A synthetic extract.py output over the real t0_s0 bank, for end-to-end tests of fit.py without a model.

Every prompt of the chosen splits (default: the extraction split, as extract.py) gets one activation per site:

    h = offset + curve(t) @ B.T + PLANT_CHOICE * s * c + noise

where t = log10(days) of the prompt's stated horizon (0 for no-horizon prompts), curve(t) = (cos 0.6t, sin 0.6t, 0.3t)
in a random 3-dimensional subspace B, c is a unit "choice" direction orthogonal to B, and s = +1 if the baseline
chose far, -1 if near. Baseline choices: the correct option where there is one, else alternating.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pbank.records import load_prompts, prompts_sha256, write_jsonl
from steer.activations import array_name, file_sha256

BANK = Path(__file__).resolve().parents[2] / "prompt_bank" / "data" / "t0_s0"
D = 24
PLANT_CHOICE = 0.5
NOISE = 0.05


def make_activations(directory: Path, layers=(15,), seed: int = 0, splits=("extraction",)) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True)
    records = load_prompts(BANK)
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.standard_normal((D, 4)))
    B, c = basis[:, :3], basis[:, 3]
    offset = rng.standard_normal(D)
    from steer.fitting import log10_days
    index, rows = [], []
    for r in (r for r in records if r.split in splits):
        t = log10_days(r.horizon_grid_value) if r.horizon is not None else 0.0
        chosen = r.correct_option or ("far" if len(index) % 2 else "near")
        index.append({"row": len(index), "uid": r.uid, "choice_label": "a", "chosen_option": chosen,
                      "label_token": " a", "prelabel_token": ":", "label_index": 3, "seq_len": 100,
                      "replay_label_margin": 1.0, "replay_predicts_label": True})
        rows.append((t, 1.0 if chosen == "far" else -1.0))
    t = np.array([x[0] for x in rows])
    s = np.array([x[1] for x in rows])
    curve = np.stack([np.cos(0.6 * t), np.sin(0.6 * t), 0.3 * t], axis=1)
    arrays = {}
    for layer in layers:
        for site in ("label", "prelabel"):
            X = offset + curve @ B.T + PLANT_CHOICE * s[:, None] * c + NOISE * rng.standard_normal((len(t), D))
            name = array_name(layer, site)
            np.save(directory / name, X.astype(np.float32))
            arrays[name] = {"sha256": file_sha256(directory / name), "shape": [len(t), D]}
    write_jsonl(directory / "index.jsonl", index)
    write_jsonl(directory / "skipped.jsonl", [])
    meta = {"settings": {"bank_prompts_sha256": prompts_sha256(records), "layers": list(layers)},
            "run_settings": {"model": "synthetic", "thinking": False}, "arrays": arrays}
    (directory / "extract.json").write_text(json.dumps(meta, indent=2) + "\n")
    return directory
