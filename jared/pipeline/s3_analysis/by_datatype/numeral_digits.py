"""Is the shared numeral code digit-structured? (post hoc, 2026-11-03)

unit_number_rsa.py: at the prompt end, renderings that share a numeral ("60 minutes",
"60 months") sit closer than gap-matched pairs (ratio ~0.8 at every layer). Levy & Geva
(2410.11781) find a per-digit circular code in LLMs. If the numeral code is digit-
structured, pairs that share the digit count, the leading digit or the last digit should
also sit closer, with the duration gap, the unit and the numeral itself held fixed.

Same E9 data and cross-validated distances as unit_number_rsa. Pairs regressed on
unit_number_rsa's terms (gap bins, same unit, same numeral, |Δ log numeral|) plus same
digit count, same leading digit and same last digit. Coefficients in units of the mean
pair distance; 95% CIs over contexts (each half resampled within itself).

    python analysis/numeral_digits.py /home/Jared/SPAR/results/horizon OUT_DIR
    -> OUT_DIR/numeral_digits.{json,txt}
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import unit_number_rsa as rsa  # noqa: E402

LAYERS = (3, 9, 15, 21, 27, 33)
BOOT = 40
DIGIT_TERMS = ("same digit count", "same leading digit", "same last digit")


def digit_design(table):
    """Per pair: same digit count, leading digit, last digit (pairs in unit_number_rsa's order)."""
    rows = []
    for i, j in combinations(range(len(table)), 2):
        a, b = table[i]["numeral"], table[j]["numeral"]
        rows.append([len(a) == len(b), a[0] == b[0], a[-1] == b[-1]])
    return np.array(rows, dtype=float)


def main(root, out_dir):
    root, out_dir = Path(root), Path(out_dir)
    rows, rendering, context, table = rsa.renderings(root)
    pairs, X, _ = rsa.pair_design(table)
    D = digit_design(table)
    design = np.column_stack([X, D])
    terms = list(rsa.TERMS) + list(DIGIT_TERMS)
    activations = np.load(root / rsa.RUN / "activations.npz")["newline"]
    rng = np.random.default_rng(0)
    result = {"pair counts": {t: int(D[:, k].sum()) for k, t in enumerate(DIGIT_TERMS)}, "layers": {}}
    for layer in tqdm(LAYERS):
        Y = activations[rows, layer].astype(np.float64)
        Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-6)
        M = rsa.context_cell_means(Y, rendering, context, len(table))
        coef, r2 = rsa.regression(rsa.cv_distances(M, pairs, rng), design)
        boot = np.array([rsa.regression(rsa.cv_distances(M, pairs, rng, resample=True), design)[0] for _ in range(BOOT)])
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        result["layers"][layer] = {"r2": r2, "coef": {t: [float(c), float(l), float(h)] for t, c, l, h in zip(terms, coef, lo, hi)}}
    (out_dir / "numeral_digits.json").write_text(json.dumps(result, indent=1))
    show = ("same unit", "same numeral", "|Δ log numeral|") + DIGIT_TERMS
    lines = ["Suffix newline, E9 renderings: pair distance (1 = mean pair) regressed on gap bins + the terms below; 95% CI over contexts",
             f"pairs sharing: {result['pair counts']}", "",
             "layer  R²    " + "  ".join(f"{t:>24s}" for t in show)]
    for layer, r in result["layers"].items():
        lines.append(f"L{layer:<4d} {r['r2']:.2f}  " + "  ".join(f"{rsa.fmt(r['coef'][t]):>24s}" for t in show))
    text = "\n".join(lines)
    (out_dir / "numeral_digits.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main(*sys.argv[1:3])
