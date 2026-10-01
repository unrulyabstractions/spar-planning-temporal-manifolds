"""Post-analysis 3: the horizon signal inside the think block.

Two questions on a thinking-on E0 run, from the synced npz and answers.json:
1. Is there a readable horizon representation during reasoning, and is it
   the suffix direction? Fits a probe per layer on the mean of the kept think
   columns and on the pooled think tokens, and reports CV R^2, the cosine
   with the suffix-fit direction, and how well the suffix probe reads the
   think mean.
2. Does the dense-track reading change where the model commits to an answer?
   Finds the first commit phrase in each think text, compares the reading in
   the 200 tokens before and after it, and relates each prompt's drift to its
   commit point and its horizon.

    python analysis/think_probe.py results/horizon_v3_think  -> <root>/analysis/think_probe.json
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, wilcoxon
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from spar_horizon.geometry import horizon_probe  # noqa: E402
from spar_horizon.runs import load_activations  # noqa: E402

COMMIT = re.compile(r"(answer (is|should be|would be)|i('ll| will| would)? (choose|go with|pick)|"
                    r"(choose|pick|go with) (option )?[ab]\)|the (best|better) (option|choice|investment) is [ab]\)|"
                    r"final answer)", re.I)
WINDOW = 200


def think_direction(z):
    """Per layer: R^2 of a probe on the think-column mean and on pooled think
    tokens, cosine with the suffix probe direction, and |rho| of the suffix
    probe read on the think mean."""
    h = z["horizons"]; has = ~np.isnan(h); lh = np.log10(h[has])
    a, b = z["regions"]["think"]
    rows = []
    for X in z["activations"]:
        mean = X[has, a:b].mean(1)
        r2, d_think, _ = horizon_probe(mean, lh)
        _, d_suf, m_suf = horizon_probe(X[has, 1], lh)
        r2_pool = horizon_probe(X[has, a:b].reshape(-1, X.shape[2]), np.repeat(lh, b - a))[0]
        rows.append({"r2_mean": float(r2), "r2_pooled": float(r2_pool), "cos_suffix": float(d_think @ d_suf),
                     "suffix_probe_rho": float(abs(spearmanr(m_suf.predict(mean), lh)[0]))})
    return rows


def commit_points(answers, tokenizer):
    """Per prompt: (fraction of think length, token index, phrase) of the first
    commit phrase, or None."""
    out = []
    for rec in answers:
        t = rec.get("think_text", "")
        m = COMMIT.search(t)
        if not m:
            out.append(None)
            continue
        out.append((m.start() / max(len(t), 1), len(tokenizer.encode(t[:m.start()], add_special_tokens=False)),
                    m.group(0)))
    return out


def commit_analysis(z, commits):
    d, tl, f, h = z["dense_track"], z["think_len"], z["forced_mask"], z["horizons"]
    found = [i for i, c in enumerate(commits) if c]
    before, after, lh = [], [], []
    for i in found:
        k = commits[i][1]
        if np.isnan(h[i]) or k < WINDOW // 2 or k + WINDOW // 2 > tl[i]:
            continue
        before.append(np.nanmean(d[i, max(0, k - WINDOW):k]))
        after.append(np.nanmean(d[i, k:min(tl[i], k + WINDOW)]))
        lh.append(np.log10(h[i]))
    before, after, lh = map(np.array, (before, after, lh))
    slopes = np.array([np.polyfit(np.linspace(0, 1, n), d[i, :n], 1)[0] for i, n in enumerate(tl)])
    has = ~np.isnan(h)
    return {"n_found": len(found), "n_forced_found": int(f[found].sum()),
            "commit_fraction_median": float(np.median([commits[i][0] for i in found])),
            "n_windowed": len(lh),
            "rho_before": float(spearmanr(before, lh)[0]), "rho_after": float(spearmanr(after, lh)[0]),
            "slope_before": float(np.polyfit(lh, before, 1)[0]), "slope_after": float(np.polyfit(lh, after, 1)[0]),
            "abs_err_before": float(np.abs(before - lh).mean()), "abs_err_after": float(np.abs(after - lh).mean()),
            "wilcoxon_p": float(wilcoxon(np.abs(before - lh), np.abs(after - lh)).pvalue),
            "drift_mean": float(slopes.mean()), "drift_nonforced": float(slopes[~f].mean()),
            "drift_vs_commit_rho": float(spearmanr(slopes[found], [commits[i][0] for i in found])[0]),
            "drift_vs_horizon_rho": float(spearmanr(slopes[has], np.log10(h[has]))[0])}


def run(root):
    root = Path(root)
    d = sorted(root.glob("exp0_*_think-on"))[-1]
    z = load_activations(d / "activations.npz")
    manifest = json.loads((d / "manifest.json").read_text())
    answers = json.loads((d / "answers.json").read_text())
    layers = think_direction(z)
    best = int(np.argmax([r["r2_mean"] for r in layers]))
    print(f"think-mean probe: best R^2 {layers[best]['r2_mean']:.2f} at L{best}, cos with suffix direction "
          f"{layers[best]['cos_suffix']:+.2f}; suffix probe reads the think mean at |rho| "
          f"{np.nanmax([r['suffix_probe_rho'] for r in layers]):.2f} at best")
    tokenizer = AutoTokenizer.from_pretrained(manifest["settings"]["model"])
    commits = commit_analysis(z, commit_points(answers, tokenizer))
    print(f"commit phrase in {commits['n_found']}/{len(answers)} prompts at median fraction "
          f"{commits['commit_fraction_median']:.2f}; reading rho before {commits['rho_before']:+.2f} / after "
          f"{commits['rho_after']:+.2f} (p={commits['wilcoxon_p']:.2f}); drift vs horizon rho "
          f"{commits['drift_vs_horizon_rho']:+.2f}, vs commit point {commits['drift_vs_commit_rho']:+.2f}")
    out = root / "analysis" / "think_probe.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"run": d.name, "layers": layers, "best_layer": best, "commit": commits}, indent=1))
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "results/horizon")
