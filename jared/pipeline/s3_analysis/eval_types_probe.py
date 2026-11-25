"""Does a horizon probe trained on the full bank read the horizon on the eval types?

    python -m pipeline.s3_analysis.eval_types_probe $RESULTS/exp0_Qwen3-8B_think-on_full \
        $RESULTS/evaltypes_Qwen3-8B_think-on

A ridge probe for log10 horizon is fitted on the full bank (Qwen3-8B, thinking on) per layer and
position, with the penalty chosen by inner CV that holds out whole horizons. It is then applied,
unchanged, to every eval-type row that has a horizon. Both runs save the same 17 positions
(suffix, think@0.25/0.5/1, pre, prelabel, answer), so a position index means the same token.

Per type and cell: Spearman of probe vs true log horizon, mean absolute error and mean bias in
decades. The reference is the probe's own 5-fold grouped CV on the full bank. Writes
`probe_transfer.json` and `probe_transfer.md` into the eval-type run directory.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

from spar_horizon.probes import fit_ridge

LAYERS = (14, 18, 22, 26)
POSITIONS = {"suffix_end": 4, "think_end": 7, "prelabel": 10}   # indices into the 17 saved tokens


def bank_data(bank_dir, layer, pos):
    z = np.load(Path(bank_dir) / "activations.npz", mmap_mode="r")
    keep = z["kept"] & np.isfinite(z["horizons"])
    X = np.asarray(z[f"L{layer}"][:, pos], dtype=np.float32)[keep]
    return X, np.log10(z["horizons"][keep])


def bank_cv(X, y):
    """Grouped 5-fold CV on the bank: Spearman and mean absolute error of held-out horizons."""
    pred = np.empty_like(y)
    for train, test in GroupKFold(5).split(X, y, y):
        pred[test] = fit_ridge(X[train], y[train], y[train]).predict(X[test])
    return float(spearmanr(pred, y)[0]), float(np.mean(np.abs(pred - y)))


def eval_rows(run_dir):
    """{type: (rows with a horizon, [their activation arrays by (layer, pos)])}."""
    out = {}
    for rows_path in sorted(Path(run_dir).glob("*/rows.json")):
        rows = [r for r in json.loads(rows_path.read_text()) if r.get("horizon") and r.get("act_file")]
        if rows:
            out[rows_path.parent.name] = rows
    return out


def eval_activations(type_dir, rows, layer, pos):
    files = {}
    X = []
    for r in rows:
        if r["act_file"] not in files:
            files[r["act_file"]] = np.load(type_dir / r["act_file"])
        X.append(files[r["act_file"]][f"L{layer}"][r["act_row"], pos])
    return np.asarray(X, dtype=np.float32)


def summarize(pred, y):
    if len(y) < 3:
        return None
    return {"n": len(y), "spearman": round(float(spearmanr(pred, y)[0]), 3),
            "mae": round(float(np.mean(np.abs(pred - y))), 3), "bias": round(float(np.mean(pred - y)), 3)}


def main(bank_dir, run_dir):
    run_dir = Path(run_dir)
    types = eval_rows(run_dir)
    result = {"bank_cv": {}, "types": {t: {} for t in types}, "predictions": {t: {} for t in types}}
    for layer in LAYERS:
        for name, pos in POSITIONS.items():
            cell = f"L{layer}_{name}"
            X, y = bank_data(bank_dir, layer, pos)
            rho, mae = bank_cv(X, y)
            result["bank_cv"][cell] = {"spearman": round(rho, 3), "mae": round(mae, 3)}
            probe = fit_ridge(X, y, y)
            for t, rows in types.items():
                Xe = eval_activations(run_dir / t, rows, layer, pos)
                pred = probe.predict(Xe)
                result["types"][t][cell] = summarize(pred, np.log10([r["horizon"] for r in rows]))
                result["predictions"][t][cell] = [round(float(p), 3) for p in pred]
            print(cell, "bank", result["bank_cv"][cell], flush=True)
    for t, rows in types.items():
        result["predictions"][t]["sample_uid"] = [r["sample_uid"] for r in rows]
    (run_dir / "probe_transfer.json").write_text(json.dumps(result, indent=1))
    (run_dir / "probe_transfer.md").write_text(markdown(result))
    print(markdown(result))


def markdown(result):
    cells = list(result["bank_cv"])
    head = "| type | " + " | ".join(cells) + " |\n|---|" + "---|" * len(cells) + "\n"
    bank = "| full bank (CV) | " + " | ".join(f"{v['spearman']:.2f} / {v['mae']:.2f}" for v in result["bank_cv"].values()) + " |\n"
    lines = []
    for t, by_cell in result["types"].items():
        lines.append(f"| {t} | " + " | ".join("-" if by_cell[c] is None else
                                             f"{by_cell[c]['spearman']:.2f} / {by_cell[c]['mae']:.2f} / {by_cell[c]['bias']:+.2f}"
                                             for c in cells) + " |")
    return ("Spearman / MAE (decades) [/ bias] of the full-bank probe's log-horizon readout.\n\n"
            + head + bank + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
