"""Behavioural analysis of a run -> results/<run>/ (CSV per table + summary.md).

  python scripts/evaluate.py runs/<run> [--out results]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from phr.analysis import MIN_EFFECT, choice_tables, multiturn_tables, state_table  # noqa: E402


def md(df: pd.DataFrame, digits: int = 3) -> str:
    """Markdown table without extra dependencies."""
    if not len(df):
        return "_(none)_"
    fmt = lambda v: f"{v:.{digits}g}" if isinstance(v, float) else str(v)   # noqa: E731
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    rows = ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, "|" + "---|" * len(df.columns), *rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--out", type=Path, default=Path("results"))
    a = ap.parse_args()
    res = a.out / a.run.name
    res.mkdir(parents=True, exist_ok=True)
    meta = json.loads((a.run / "capture_meta.json").read_text())
    cfg = yaml.safe_load((a.run / "variants.yaml").read_text())
    lines = [f"# Phrasing study: {a.run.name}", "",
             f"Model `{meta.get('model')}` (rev {meta.get('model_revision')}), commit {meta.get('git_commit')}, "
             f"GPU {meta.get('gpu')}. Choice readout separator {meta.get('choice_separator')!r}.", ""]
    tables = {}
    if (a.run / "choice.parquet").exists():
        ch = pd.read_parquet(a.run / "choice.parquet")
        t = choice_tables(ch)
        tables.update(t)
        per = t["choice_variants"].sort_values(["family", "mean_delta"])
        g = pd.read_parquet(a.run / "gencheck.parquet")
        ok = g.gen_short.notna()
        lines += ["## Choice protocol", "",
                  f"{len(ch)} items. Free-generation check: {ok.mean():.0%} follow the format, readout agrees on "
                  f"{(g[ok].gen_short == g[ok].readout_short).mean():.0%}. Readout mass on the two labels: "
                  f"median {ch.mass_ab.median():.3f}, min {ch.mass_ab.min():.3f}.", "",
                  "### Validity gate (canonical curve)", "", md(t["choice_gate"]), "", md(t["choice_curve"]), "",
                  f"### Variants: Δ P(short) vs reference (95% CI over scenarios). `matters` = above the null floor, "
                  f"CI excludes 0, |Δ| ≥ {MIN_EFFECT}", "", md(per.drop(columns=["above_floor"])), "",
                  "### Effective-horizon shift (log10 years; multiplier = 10^shift; < 1 acts like a shorter horizon)", "",
                  md(t["choice_shifts"]), "", "### Contrasts", "", md(t["choice_contrasts"]), "",
                  "### Layout cross: the same variant's Δ P(short) in each layout (vs that layout's canonical)", "",
                  md(t["choice_layout_cross"]), "",
                  "### Implicit vs explicit twin (Δ P(short)), by layout, determinacy and hidden number", "",
                  md(t["choice_implicit"]), ""]
    if (a.run / "state.parquet").exists():
        st = pd.read_parquet(a.run / "state.parquet")
        tables["state"] = state_table(st)
        tables["state_items"] = st[["variant", "domain", "h_text", "answer", "stated_years", "h_years"]]
        lines += ["## State protocol (the model names the horizon)", "", md(tables["state"]), ""]
    if (a.run / "multiturn.parquet").exists():
        mt = pd.read_parquet(a.run / "multiturn.parquet")
        det = {i["id"]: i["determinacy"] for i in cfg["choice"]["implicit"]}
        t = multiturn_tables(mt, cfg["multiturn"]["cue_step"], det)
        tables.update(t)
        lines += ["## Multi-turn protocol", "", md(t["multiturn_adherence"]), "",
                  "Differences in log10 years of the plan's horizons vs the reference conversation "
                  "(multiplier = 10^mean; 95% CI over scenarios).", "", md(t["multiturn_comparisons"]), ""]
    for name, df in tables.items():
        df.to_csv(res / f"{name}.csv", index=False)
    (res / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
