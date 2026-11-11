"""Derive the numbers behind the prompt-type figures and write them to
figures/prompt_stats.json.

Needs pandas + pyarrow (for Alan's and Augusto's parquet files), which none of
the conda envs has, so run it with uv:

    uv run --no-project --with pandas --with pyarrow python PromptDefinitions/scripts/extract_prompt_stats.py

The figures themselves (make_figures.py) only need the JSON, numpy and
matplotlib, so they run on the intel_ai env.
"""
import json
import math
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]          # spar-planning-temporal-manifolds/
SPAR = ROOT.parent
ALAN = ROOT / "alan" / "data" / "prompts"
OUT = ROOT / "PromptDefinitions" / "figures" / "prompt_stats.json"

sys.path.insert(0, str(ROOT / "jared"))
from spar_horizon.full_bank import build_full_prompts  # noqa: E402
from spar_horizon.prompts import build_prompts  # noqa: E402

DAYS_PER_YEAR = 365.25

# Prior paper GEOMETRY_CFG horizons (temporal-awareness/src/intertemporal/data/default_datasets.py:128-146).
# Copied rather than imported: that package pulls in torch.
PRIOR_GEOMETRY_HORIZONS = {
    "1 second": 1 / (DAYS_PER_YEAR * 86400), "1 hour": 1 / (DAYS_PER_YEAR * 24), "1 day": 1 / DAYS_PER_YEAR,
    "1 week": 7 / DAYS_PER_YEAR, "1 month": 1 / 12, "2 months": 2 / 12, "6 months": 0.5, "1 year": 1,
    "3 years": 3, "5 years": 5, "1 decade": 10, "3 decades": 30, "5 decades": 50,
    "1 century": 100, "2 centuries": 200, "5 centuries": 500,
}


def singular(unit):
    """'centuries' -> 'century', 'years' -> 'year'."""
    if unit.endswith("ies"):
        return unit[:-3] + "y"
    return unit.rstrip("s")


def regime(horizon, near_delay, far_delay):
    """Same definition as jared/spar_horizon/full_bank.py:284."""
    if horizon is None or (isinstance(horizon, float) and math.isnan(horizon)):
        return "null"
    if horizon < near_delay:
        return "none"
    if horizon < far_delay:
        return "near"
    return "both"


def alan_frame_stats(df):
    regimes = Counter(regime(h, s, l) for h, s, l in
                      zip(df.horizon_years, df.short_delay_years, df.long_delay_years))
    return {"n": len(df), "regimes": dict(regimes), "near_first": float(df.short_first.mean())}


def grid(df):
    rows = df.dropna(subset=["horizon_years"]).drop_duplicates("horizon_text")
    return sorted(([t, y] for t, y in zip(rows.horizon_text, rows.horizon_years)), key=lambda r: r[1])


def main():
    stats = {"datasets": {}, "grids": {}}

    # Jared, starter bank: near option always first.
    starter = build_prompts(seed=0)
    stats["datasets"]["Jared starter (J1)"] = {
        "n": len(starter),
        "regimes": dict(Counter(regime(r["horizon"], r["near_delay_years"], r["far_delay_years"]) for r in starter)),
        "near_first": 1.0,
    }
    stats["grids"]["Jared starter, 16"] = sorted({r["horizon"] for r in starter if r["horizon"] is not None})

    # Jared, full bank (seed 0, n 500, as run).
    full = build_full_prompts(seed=0, n=500)
    stats["datasets"]["Jared full bank (J2)"] = {
        "n": len(full),
        "regimes": dict(Counter(r["regime"] for r in full)),
        "near_first": sum(r["short_first"] for r in full) / len(full),
    }
    stats["grids"]["Jared full bank, 48"] = sorted({r["horizon"] for r in full if r["horizon"] is not None})
    stats["full_bank_units"] = [[singular(r["horizon_text"].split()[1]), r["horizon"]]
                                for r in full if r["horizon"] is not None]

    # Alan.
    canonical = pd.read_parquet(ALAN / "investment_n2000_s0.parquet")
    extended = pd.read_parquet(ALAN / "investment_extended_n2000_s0.parquet")
    phrasings = pd.read_parquet(ALAN / "investment_phrasings_n2000_s0.parquet")
    units = pd.read_parquet(ALAN / "investment_units_paired_n2000_s1.parquet")
    matrix = pd.read_parquet(ALAN / "matrix" / "matrix_s0.parquet")
    controls = pd.read_parquet(ALAN / "matrix" / "controls_s0.parquet")
    stats["datasets"]["Alan replication (A1)"] = alan_frame_stats(canonical)
    stats["datasets"]["Alan extended (A2)"] = alan_frame_stats(extended)
    stats["datasets"]["Alan phrasings (A3)"] = alan_frame_stats(phrasings)
    stats["datasets"]["Alan unit rewrites (A4)"] = alan_frame_stats(units)
    stats["datasets"]["Alan matrix (A5)"] = alan_frame_stats(matrix)
    stats["datasets"]["Alan controls (A6)"] = alan_frame_stats(controls[controls.condition != "factual"])
    stats["grids"]["Alan standard, 17"] = [y for _, y in grid(canonical)]
    stats["grids"]["Alan extended, 25"] = [y for _, y in grid(extended)]
    stats["grids"]["Alan matrix, 12"] = [y for _, y in grid(matrix)]
    stats["matrix_heldout"] = sorted(set(matrix.loc[matrix.horizon_heldout == True, "horizon_years"]))  # noqa: E712
    stats["grids"]["Prior paper geometry, 16"] = sorted(PRIOR_GEOMETRY_HORIZONS.values())

    # Option-order balance within a config: is short_first constant per config?
    per_config = matrix.groupby("config_id").short_first.nunique()
    stats["matrix_order_constant_per_config"] = bool((per_config == 1).all())

    # Alan's paired unit rewrites: canonical horizon vs rendered unit.
    u = units.dropna(subset=["horizon_years"])
    stats["units_paired"] = [[singular(unit), canon_years]
                             for unit, canon_years in zip(u.horizon_unit, u.horizon_years)]

    # Option delay ranges (the regime boundaries).
    stats["delays"] = {
        "Alan / full bank short": [7 / DAYS_PER_YEAR, 1.0],
        "Alan / full bank long": [2.0, 50.0],
        "Starter near": [1 / 12, 0.5],
        "Starter far": [5.0, 10.0],
        "Prior short": [1 / DAYS_PER_YEAR, 20.0],
        "Prior long": [1.0, 100.0],
    }

    # Augusto: distractor gap log10(D / H) per family.
    rad = pd.read_parquet(ROOT / "relevance_aware_decoder" / "data" / "prompts_s0.parquet")
    dis = rad[rad.condition == "distractor"]
    stats["datasets"]["Augusto distractors (R1)"] = alan_frame_stats(dis)
    stats["distractor_gaps"] = {fam: g.log_gap.round(4).tolist() for fam, g in dis.groupby("family")}
    stats["distractor_D_years"] = {fam: sorted(set(g.distractor_years.round(6))) for fam, g in dis.groupby("family")}

    # Prior paper CAA pairs: share of pairs with the immediate option as (A).
    raw = SPAR / "temporal-awareness" / "data" / "raw"
    caa = {}
    for rel in ["temporal_scope/temporal_scope_implicit_expanded_1000.json",
                "temporal_scope_AB_randomized/temporal_scope_implicit_expanded_1000.json",
                "temporal_scope/temporal_scope_explicit_expanded_500.json",
                "temporal_scope_AB_randomized/temporal_scope_explicit_expanded_500.json"]:
        data = json.loads((raw / rel).read_text())
        pairs = data["pairs"] if isinstance(data, dict) else data
        caa[rel] = sum(p["immediate"].strip().startswith("(A)") for p in pairs) / len(pairs)
    stats["caa_immediate_is_A"] = caa

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(stats, indent=1))
    print(f"wrote {OUT}")
    for name, d in stats["datasets"].items():
        print(f"{name:28s} n={d['n']:5d} near_first={d['near_first']:.2f} {d['regimes']}")
    print("matrix order constant per config:", stats["matrix_order_constant_per_config"])
    print("CAA immediate=(A):", caa)


if __name__ == "__main__":
    main()
