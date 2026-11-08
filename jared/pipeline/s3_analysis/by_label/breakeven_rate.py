"""Does the choice follow the break-even reinvestment rate? (post hoc, 2026-11-03)

The coded traces (results 2026-11-03 §B) read the both-regime task as "final wealth if
the near payout is reinvested". Under that reading the relevant quantity is the annual
rate at which reinvesting the near reward until the far delay matches the far reward:

    r* = (R_far / R_near) ** (1 / (d_far - d_near)) - 1        (d in years)

A model that compares against a plausible market rate picks near when r* is low and
far when it is high. Saved answers only. Per model, bank and regime (natural closes):
near share by bins of log10 r*, and a logistic fit of chose-near on log10 r* alone, on
log10 r* plus log10 horizon / near delay (the none-regime gradient), and on the
components (log ratio, log delay gap) to see whether r* is the right combination.
Fit quality is the out-of-fold log loss (5 folds, base prompts grouped).

    python -m pipeline.s3_analysis.by_label.breakeven_rate /home/Jared/SPAR/results/horizon OUT_DIR [on|off]
    -> OUT_DIR/breakeven_rate.{json,txt,png} (thinking on), breakeven_rate_off.* (thinking off;
       there is no think block, so every row counts as a natural close)
"""
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold

from pipeline.s3_analysis.by_datatype.written_unit import run_records

warnings.filterwarnings("ignore", category=FutureWarning)
RUNS = (("Qwen3-8B", ("full", "units", "fill")), ("Qwen3-14B", ("full",)))
RATE_BINS = (-2, -1, 0, 1, 2)          # log10 r*: 1%, 10%, 100%, 1000%, 10^4 % a year
RATE_CAP = 4.0                          # log10 r*: above 10^4 per year every reading is "far wins"; the fill's
                                        # delays of seconds give exponents past float range otherwise
MODELS = {"r*": ["log_rate"], "r* + horizon/near delay": ["log_rate", "log_h_near"],
          "ratio + gap": ["log_ratio", "log_gap"], "ratio + gap + horizon/near delay": ["log_ratio", "log_gap", "log_h_near"]}


def log10_breakeven(ratio, gap):
    """log10 of (ratio ** (1 / gap) - 1), computed in log space and capped at RATE_CAP."""
    exponent = np.log(ratio) / gap
    if exponent > 20:                   # expm1(e) ~ exp(e)
        return min(exponent / np.log(10), RATE_CAP)
    return min(float(np.log10(np.expm1(exponent))), RATE_CAP)


def rows_of(root, model, bank, mode="on"):
    run = f"exp0_{model}_think-{mode}_{bank}"
    records = run_records(root, run)
    out = []
    for a in json.loads((root / run / "answers.json").read_text()):
        if a.get("horizon") is None or "think_len" not in a or a.get("anchor") or a["forced"]:
            continue
        r = records[a["sample_uid"]]
        ratio = r["rewards"][1] / r["rewards"][0]
        gap = r["far_delay_years"] - r["near_delay_years"]
        out.append({"bank": bank, "regime": a["regime"], "near": int(a["label"] == a["near_label"]),
                    "group": f'{bank}|{a.get("pair_id", a["sample_uid"])}',
                    "log_rate": log10_breakeven(ratio, gap), "log_ratio": float(np.log10(ratio)),
                    "log_gap": float(np.log10(gap)), "log_h_near": float(np.log10(a["horizon"] / r["near_delay_years"]))})
    return out


def oof_log_loss(rows, columns):
    X = np.array([[r[c] for c in columns] for r in rows])
    y = np.array([r["near"] for r in rows])
    groups = np.array([r["group"] for r in rows])
    pred = np.zeros(len(y))
    for train, test in GroupKFold(n_splits=5).split(X, y, groups):
        mu, sd = X[train].mean(0), X[train].std(0) + 1e-9
        model = LogisticRegression(C=10.0, max_iter=2000).fit((X[train] - mu) / sd, y[train])
        pred[test] = model.predict_proba((X[test] - mu) / sd)[:, 1]
    full = LogisticRegression(C=10.0, max_iter=2000).fit((X - X.mean(0)) / (X.std(0) + 1e-9), y)
    return float(log_loss(y, pred)), [float(c) for c in full.coef_[0]]


def binned(rows):
    bins = np.digitize([r["log_rate"] for r in rows], RATE_BINS)
    return {int(b): [int((bins == b).sum()), float(np.mean([r["near"] for r, k in zip(rows, bins) if k == b]))]
            for b in np.unique(bins)}


def analyse(root, mode="on"):
    result = {}
    for model, banks in RUNS:
        rows = [r for bank in banks for r in rows_of(root, model, bank, mode)]
        for regime in ("none", "both"):
            for scope, chosen in (("all banks", banks), ("old banks", tuple(b for b in banks if b != "fill"))):
                if scope == "old banks" and chosen == banks:
                    continue
                sel = [r for r in rows if r["regime"] == regime and r["bank"] in chosen]
                y = np.array([r["near"] for r in sel])
                if len(sel) < 30 or y.mean() in (0, 1):
                    continue
                entry = {"n": len(sel), "near share": float(y.mean()),
                         "null log loss": float(log_loss(y, np.full(len(y), y.mean()))), "bins": binned(sel), "fits": {}}
                for name, columns in MODELS.items():
                    loss, coef = oof_log_loss(sel, columns)
                    entry["fits"][name] = {"oof log loss": loss, "coef (per SD)": dict(zip(columns, coef))}
                result[f"{model} {regime} {scope}"] = entry
    return result


def plot(rows_by_key, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (key, rows), color in zip(rows_by_key.items(), ("#2a78d6", "#eb6834", "#1baf7a", "#7b4fd6")):
        x = np.array([r["log_rate"] for r in rows])
        y = np.array([r["near"] for r in rows])
        edges = np.quantile(x, np.linspace(0, 1, 8))
        idx = np.clip(np.digitize(x, edges[1:-1]), 0, 6)
        centers = [x[idx == k].mean() for k in range(7) if (idx == k).any()]
        shares = [y[idx == k].mean() for k in range(7) if (idx == k).any()]
        ax.plot(centers, shares, "-o", color=color, label=f"{key} (n {len(rows)})")
    ax.set_xlabel("log10 break-even annual rate r* (near reinvested matches far)")
    ax.set_ylabel("share choosing near")
    ax.axvline(np.log10(0.07), color="gray", ls=":", lw=1)
    ax.text(np.log10(0.07), 1.02, "7%/yr", fontsize=8, ha="center")
    ax.legend(fontsize=8)
    ax.set_title("Both regime, natural closes: near share by break-even rate (septiles)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main(root, out_dir, mode="on"):
    root, out_dir = Path(root), Path(out_dir)
    stem = "breakeven_rate" if mode == "on" else f"breakeven_rate_{mode}"
    result = analyse(root, mode)
    (out_dir / f"{stem}.json").write_text(json.dumps(result, indent=1))
    lines = [f"Thinking {mode}. Break-even reinvestment rate r* and the choice (natural closes). Bins of log10 r*: "
             f"edges {RATE_BINS} -> bin: [n, near share]. OOF log loss (lower is better; null = base rate).", ""]
    for key, e in result.items():
        lines.append(f"{key}: n {e['n']}, near share {e['near share']:.2f}, null log loss {e['null log loss']:.3f}")
        lines.append("   bins " + " ".join(f"{b}:{n}/{s:.2f}" for b, (n, s) in e["bins"].items()))
        for name, f in e["fits"].items():
            coefs = ", ".join(f"{c} {v:+.2f}" for c, v in f["coef (per SD)"].items())
            lines.append(f"   {name:34s} log loss {f['oof log loss']:.3f}   ({coefs})")
        lines.append("")
    text = "\n".join(lines)
    (out_dir / f"{stem}.txt").write_text(text)
    rows_by_key = {}
    for model, banks in RUNS:
        rows = [r for bank in banks for r in rows_of(root, model, bank, mode) if r["regime"] == "both"]
        rows_by_key[f"{model} old banks"] = [r for r in rows if r["bank"] != "fill"]
        if "fill" in banks:
            rows_by_key[f"{model} fill"] = [r for r in rows if r["bank"] == "fill"]
    plot(rows_by_key, out_dir / f"{stem}.png")
    print(text)


if __name__ == "__main__":
    main(*sys.argv[1:4])
