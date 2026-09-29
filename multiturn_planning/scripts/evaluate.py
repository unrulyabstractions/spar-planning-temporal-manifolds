"""Run analyses A-D on a capture run and write CSVs + summary.md to results/<run name>/.

    python scripts/evaluate.py runs/<name> [--layers 16,24,32] [--quick]

--quick: pre-reply positions P0, P4, P8 and U only (fast look at a pilot).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mtp import analysis as an  # noqa: E402

PRE = [f"P{i}" for i in range(9)]


def best(df, col, by=None, n=1):
    d = df.sort_values(col, ascending=False)
    return d.groupby(by).head(n) if by else d.head(n)


def fmt(df, cols):
    return df[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--layers", default=None, help="comma-separated residual layers (default: all captured)")
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    run = Path(a.run)
    res = run.parent.parent / "results" / run.name
    res.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df, acts, valid, meta = an.load_run(run)
    layers = [int(x) for x in a.layers.split(",")] if a.layers else meta["layers"]
    li = [meta["layers"].index(l) for l in layers]
    pre = ["P0", "P4", "P8", "U"] if a.quick else PRE + ["U"]
    convs = an.clean_conversations(df)
    log = [f"# Multi-turn planning: {run.name}", "",
           f"model `{meta['model']}` · {df.conv_id.nunique()} conversations · clean (all 5 steps parsed and "
           f"numbered): {len(convs)} · layers {layers}", ""]

    beh = an.behavior(df, convs)
    json.dump(beh, open(res / "behavior.json", "w"), indent=1)
    log += ["## Behavior", "", *[f"- {k}: {v:.3f}" if isinstance(v, float) else f"- {k}: {v}" for k, v in beh.items()], ""]

    A = an.analysis_a(df, acts, valid, meta, pre + ["E"], li)
    A.to_csv(res / "A_persistence.csv", index=False)
    b1 = best(A[A.turn == 1], "r2")
    cell = A[(A.layer == b1.layer.iloc[0]) & (A.pos == b1.pos.iloc[0])]
    log += ["## A · persistence of H_target (probe from turn 1, tested at turn t)", "",
            f"best turn-1 cell: layer {b1.layer.iloc[0]}, {b1.pos.iloc[0]}; that cell across turns:", "",
            "```", fmt(cell, ["turn", "r2", "rho", "within2x", "r2_same_turn", "pc1_rho"]), "```", ""]

    B = pd.concat([an.analysis_b(df, acts, valid, meta, pre + ["H", "V"], li, convs, c) for c in ("horizon", "none")])
    B.to_csv(res / "B_next_step.csv", index=False)
    log += ["## B / D · the step about to be written (text baseline vs activations)", "",
            "```", fmt(best(B[~B.pos.isin(["H", "V"])], "delta_r2", by="condition", n=3),
                       ["condition", "layer", "pos", "n", "r2_text", "r2_act", "r2_text_plus_act", "delta_r2",
                        "delta_r2_shuffled", "within2x_text", "within2x_text_plus_act"]), "```", "",
            "reference positions inside the reply (H = 'Time horizon:', V = the value itself):", "",
            "```", fmt(best(B[B.pos.isin(["H", "V"])], "r2_act", by=["condition", "pos"]),
                       ["condition", "pos", "layer", "r2_act", "delta_r2"]), "```", ""]

    D = an.analysis_d_transfer(df, acts, meta, pre + ["H"], li, convs)
    D.to_csv(res / "D_transfer.csv", index=False)
    log += ["## D · probe trained on horizon conversations, applied to no-horizon ones", "",
            "```", fmt(best(D, "r2", n=3), ["layer", "pos", "r2", "rho", "within2x", "n"]), "```", ""]

    C = an.analysis_c(df, acts, meta, pre, li, convs)
    C.to_csv(res / "C_first_turn.csv", index=False)
    log += ["## C · turn-1 activations and later steps, beyond H_target", "",
            "```", fmt(best(C, "delta_r2", by="step"),
                       ["step", "layer", "pos", "r2_target_only", "r2_target_plus_act", "delta_r2"]), "```", ""]

    log.append(f"_evaluate: {time.time() - t0:.0f}s_")
    (res / "summary.md").write_text("\n".join(log))
    print("\n".join(log))
