#!/usr/bin/env python
"""Dense single-quantity set: one fixed option configuration and label order, ~N distinct integers log-spaced from 1 to
MAX placed in ONE slot, everything else fixed.

  --vary horizon       (default) the integers are the horizon in UNIT; prompts differ only in the CONSTRAINT line
  --vary short_reward  the integers are the short option's reward, the horizon fixed at --horizon; prompts differ only
                       in the short option's line. With the default grid this reuses the horizon set's exact integers,
                       the non-temporal specificity control.
  --vary long_reward   the same for the long option's reward (crosses the fixed short reward: a magnitude-comparison
                       choice switch)

  gen_dense_horizons.py OUT.parquet [--vary horizon|short_reward|long_reward] [--n 1500] [--unit days] [--max 36525] [--horizon "5 years"]
      [--short-reward 10000 --short-delay "1 year" --long-reward 50000 --long-delay "10 years"] [--long-first] [--no-reward-commas]
      [--constraint-first] [--units days:467,weeks:467,... | --matched-base months --units days,weeks,months]
"""
import argparse, json
from pathlib import Path

from ptm.horizons import UNIT_YEARS, Horizon, canonical_unit
from ptm.prompts import PromptFormat, dense_integer_grid, fixed_scenario, to_frame

ap = argparse.ArgumentParser()
ap.add_argument("out", type=Path)
ap.add_argument("--vary", choices=["horizon", "short_reward", "long_reward"], default="horizon")
ap.add_argument("--n", type=int, default=1500); ap.add_argument("--unit", default="days"); ap.add_argument("--max", type=int, default=36525)
ap.add_argument("--units", default=None, help='pooled multi-unit horizon set, e.g. "days:467,weeks:467,months:466,years:100": '
                "per unit, that many log-spaced integers from 1 to the largest whole count within 100 years (overrides --unit/--n/--max)")
ap.add_argument("--horizon", default="5 years", help="fixed horizon when --vary short_reward")
ap.add_argument("--short-reward", type=float, default=10_000); ap.add_argument("--short-delay", default="1 year")
ap.add_argument("--long-reward", type=float, default=50_000); ap.add_argument("--long-delay", default="10 years")
ap.add_argument("--long-first", action="store_true")
ap.add_argument("--no-reward-commas", action="store_true", help='render rewards as "36525" instead of "36,525"')
ap.add_argument("--matched-base", default=None, help="matched-duration per-unit sets: --n/--max give integers in this unit; each is "
                "rendered in every unit of --units (plain list) by rounding the duration; pair_id = base value")
ap.add_argument("--render", default="plain", help="horizon renderings, comma-separated: plain | padN | decN (e.g. plain,pad4,dec2); "
                "with several, one copy of the set per rendering (single-unit --vary horizon only)")
ap.add_argument("--constraint-first", action="store_true", help="CONSTRAINT line before the options (line-position control)")
a = ap.parse_args()
values = dense_integer_grid(a.n, a.max)
common = dict(short_delay=Horizon.parse(a.short_delay), long_delay=Horizon.parse(a.long_delay),
              short_first=not a.long_first, reward_commas=not a.no_reward_commas,
              fmt=PromptFormat(name="formatted_v1_constraint_first", constraint_first=True) if a.constraint_first else None)
if a.vary == "horizon" and a.matched_base:
    base = canonical_unit(a.matched_base); samples = []
    rd = a.render
    if "," in rd or not (rd == "plain" or rd.startswith("dec")):
        raise SystemExit("--matched-base takes one --render: plain (round to integers) or decN (round to N decimals)")
    for spec in a.units.split(","):
        u = canonical_unit(spec)
        if rd == "plain":
            vu = [max(1, int(round(v * UNIT_YEARS[base] / UNIT_YEARS[u]))) for v in values]
        else:                                   # the stored value is the rendered one, so horizon_years matches the text
            vu = [round(v * UNIT_YEARS[base] / UNIT_YEARS[u], int(rd[3:])) for v in values]
        if len(set(vu)) != len(vu) or min(vu) <= 0:
            raise SystemExit(f"{u}: rounding the {base} grid gives repeated or zero values; use a coarser base or more decimals")
        ss = fixed_scenario([Horizon(x, u) for x in vu], short_reward=a.short_reward, long_reward=a.long_reward,
                            uid_prefix=f"matched_{base}_{u}" + ("" if rd == "plain" else f"_{rd}"), horizon_render=rd, **common)
        for s, v in zip(ss, values):
            s.pair_id = f"{base}_{v:06d}"
        samples += ss
        err = max(abs(x * UNIT_YEARS[u] / (v * UNIT_YEARS[base]) - 1) for x, v in zip(vu, values))
        print(f"  {u}: {len(vu)} values {vu[0]}..{vu[-1]}, max relative duration error vs base {err:.4f}")
elif a.vary == "horizon" and a.units:
    samples, values = [], []
    for spec in a.units.split(","):
        u, k = spec.split(":"); u = canonical_unit(u)
        vu = dense_integer_grid(int(k), int(100 / UNIT_YEARS[u] + 1e-9))
        samples += fixed_scenario([Horizon(v, u) for v in vu], short_reward=a.short_reward, long_reward=a.long_reward,
                                  uid_prefix=f"dense_{u}", **common)
        values += vu
        print(f"  {u}: {len(vu)} values {vu[0]}..{vu[-1]}")
elif a.vary == "horizon":
    samples = []
    for rd in a.render.split(","):
        samples += fixed_scenario([Horizon(v, a.unit) for v in values], short_reward=a.short_reward, long_reward=a.long_reward,
                                  uid_prefix=f"dense_{a.unit}" + ("" if rd == "plain" else f"_{rd}"), horizon_render=rd, **common)
elif a.vary == "short_reward":
    samples = fixed_scenario(Horizon.parse(a.horizon), short_reward=[float(v) for v in values], long_reward=a.long_reward,
                             uid_prefix="dense_short_reward", **common)
else:
    samples = fixed_scenario(Horizon.parse(a.horizon), short_reward=a.short_reward, long_reward=[float(v) for v in values],
                             uid_prefix="dense_long_reward", **common)
df = to_frame(samples)
a.out.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(a.out, index=False)
a.out.with_suffix(".config.json").write_text(json.dumps(vars(a) | {"out": str(a.out), "n_values": len(values)}, indent=2))
col = "horizon_text" if a.vary == "horizon" else a.vary
print(f"wrote {len(df)} prompts to {a.out}: vary {a.vary}, values {min(values)}..{max(values)}, {df.text.nunique()} distinct texts, "
      f"{df[col].nunique()} distinct {col}, {df.horizon_years.nunique()} distinct horizon_years")
print("\n--- sample 0 ---\n" + samples[0].text + "\n--- last sample, differing line ---\n"
      + next(l for l, m in zip(samples[-1].text.split("\n"), samples[0].text.split("\n")) if l != m))
