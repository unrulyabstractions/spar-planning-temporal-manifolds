"""Post-analysis 2: does the horizon axis transfer between thinking modes?

Both E0 runs share the turn-suffix positions up to `<think>`. For every
layer and shared position, train the horizon probe on one mode and evaluate
on the other (Spearman |rho| of prediction vs log horizon), and compute the
cosine between the two probe directions. Runs on the laptop from the synced
npz files; no model needed.

    python analysis/cross_mode_transfer.py results/horizon_v1_noprefill
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from spar_horizon.geometry import horizon_probe  # noqa: E402
from spar_horizon.runs import load_activations  # noqa: E402


def load(root, mode):
    z = load_activations(root / f"exp0_Qwen3-8B_think-{mode}" / "activations.npz")
    has_h = ~np.isnan(z["horizons"])
    return z, has_h, np.log10(z["horizons"][has_h])


def shared_positions(tok_a, tok_b):
    n = 0
    for a, b in zip(tok_a, tok_b):
        if a != b:
            break
        n += 1
    return n


def main(root):
    root = Path(root)
    off, has_off, lh_off = load(root, "off")
    on, has_on, lh_on = load(root, "on")
    n_pos = shared_positions(off["tokens"], on["tokens"])
    tokens = off["tokens"][:n_pos]
    n_layers = len(off["activations"])
    print(f"shared positions: {n_pos} {tokens}")

    off_to_on = np.full((n_layers, n_pos), np.nan)
    on_to_off = np.full((n_layers, n_pos), np.nan)
    within_off = np.full((n_layers, n_pos), np.nan)
    within_on = np.full((n_layers, n_pos), np.nan)
    cosine = np.full((n_layers, n_pos), np.nan)
    for l in range(n_layers):
        for p in range(n_pos):
            X_off = off["activations"][l][has_off, p]
            X_on = on["activations"][l][has_on, p]
            if np.allclose(X_off - X_off.mean(0), 0) or np.allclose(X_on - X_on.mean(0), 0):
                continue
            r2_off, d_off, m_off = horizon_probe(X_off, lh_off)
            r2_on, d_on, m_on = horizon_probe(X_on, lh_on)
            within_off[l, p], within_on[l, p] = r2_off, r2_on
            off_to_on[l, p] = abs(spearmanr(m_off.predict(X_on), lh_on)[0])
            on_to_off[l, p] = abs(spearmanr(m_on.predict(X_off), lh_off)[0])
            cosine[l, p] = abs(float(d_off @ d_on))
        print(f"L{l:<3} off->on {np.nanmean(off_to_on[l]):.3f}  on->off {np.nanmean(on_to_off[l]):.3f}  "
              f"cos {np.nanmean(cosine[l]):.3f}  within off/on R2 {np.nanmean(within_off[l]):.2f}/{np.nanmean(within_on[l]):.2f}")

    out = root / "analysis"
    out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(4.2 * 3, 6), sharey=True, layout="constrained")
    for ax, M, title, vmin in zip(axes, (off_to_on, on_to_off, cosine),
                                  ("probe trained OFF, tested ON  |rho|",
                                   "probe trained ON, tested OFF  |rho|",
                                   "|cos| between the two probe directions"), (0, 0, 0)):
        im = ax.imshow(M, vmin=vmin, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(n_pos), [repr(t) for t in tokens], rotation=45, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("position")
    axes[0].set_ylabel("layer")
    fig.colorbar(im, ax=axes, shrink=0.6)
    fig.suptitle("Horizon axis transfer between thinking modes (Qwen3-8B, E0 suffix positions)")
    fig.savefig(out / "cross_mode_transfer.png", dpi=150)
    best = np.unravel_index(np.nanargmax(np.fmin(off_to_on, on_to_off)), off_to_on.shape)
    summary = {"shared_tokens": tokens, "off_to_on": off_to_on.tolist(), "on_to_off": on_to_off.tolist(),
               "cosine": cosine.tolist(), "within_off_r2": within_off.tolist(), "within_on_r2": within_on.tolist(),
               "best_symmetric": {"layer": int(best[0]), "pos": int(best[1]), "token": tokens[best[1]],
                                  "off_to_on": float(off_to_on[best]), "on_to_off": float(on_to_off[best]),
                                  "cosine": float(cosine[best])}}
    (out / "cross_mode_transfer.json").write_text(json.dumps(summary, indent=1))
    print(f"\nbest symmetric transfer: layer {best[0]}, token {tokens[best[1]]!r}: "
          f"off->on {off_to_on[best]:.3f}, on->off {on_to_off[best]:.3f}, cos {cosine[best]:.3f}")
    print(f"wrote {out / 'cross_mode_transfer.png'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/horizon")
