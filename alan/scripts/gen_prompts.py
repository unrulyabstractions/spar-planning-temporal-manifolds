#!/usr/bin/env python
"""Generate a prompt dataset parquet.  Usage: gen_prompts.py OUT.parquet [--n N] [--seed S] [--domains investment,climate]"""
import argparse
from pathlib import Path

from ptm.horizons import GRIDS
from ptm.prompts import DatasetConfig, generate, to_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--domains", default="investment")
    ap.add_argument("--unit-rewrite", action="store_true", help="sample horizon renderings across units (see UNIT_REWRITES)")
    ap.add_argument("--phrasings", default="0", help="comma-separated CONSTRAINT_PHRASINGS indices to sample from")
    ap.add_argument("--paired", action="store_true", help="emit every rendering/phrasing of each base sample (shared pair_id)")
    ap.add_argument("--grid", choices=sorted(GRIDS), default="standard", help="horizon grid (standard: 17 values 1 day-100 y; extended: +4 below, +4 above)")
    a = ap.parse_args()
    horizons = GRIDS[a.grid]
    cfg = DatasetConfig(n=a.n, seed=a.seed, domains=tuple(a.domains.split(",")), unit_rewrite=a.unit_rewrite, paired=a.paired,
                        phrasing_ids=tuple(int(x) for x in a.phrasings.split(",")), horizons=horizons,
                        null_horizon_fraction=1.0 / (len(horizons) + 1))
    samples = generate(cfg)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df = to_frame(samples)
    df.to_parquet(a.out, index=False)
    (a.out.with_suffix(".config.json")).write_text(cfg.to_json())
    print(f"wrote {len(df)} prompts to {a.out}")
    print(f"null-horizon: {df['horizon_text'].isna().sum()}  short_first: {df['short_first'].sum()}  bins: {df['horizon_text'].nunique()}  per-bin min/median/max: {df['horizon_text'].value_counts().agg(['min','median','max']).astype(int).tolist()}")
    if a.unit_rewrite:
        print("per (canonical, rendering) counts:", df.groupby(["horizon_canonical", "horizon_text"]).size().describe()[["min", "50%", "max"]].to_dict())
        print("units:", df["horizon_unit"].value_counts().to_dict())
    if a.paired:
        print("pairs:", df["pair_id"].nunique(), " renderings per pair:", df.groupby("pair_id").size().value_counts().sort_index().to_dict())
    if len(cfg.phrasing_ids) > 1:
        print("phrasing counts:", df["phrasing_id"].value_counts().to_dict())
    print("\n--- sample 0 ---\n" + samples[0].text)


if __name__ == "__main__":
    main()
