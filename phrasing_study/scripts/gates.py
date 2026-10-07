"""Sanity gates on a run (exit 1 if any fails). Used after the smoke slice, before the full run is started.

  python scripts/gates.py runs/<run>
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

GATES = {   # name: (threshold, description)
    "readout_mass": (0.5, "median probability mass on the two label tokens after the prefill"),
    "gencheck_format": (0.8, "free generations that write 'I choose: <label>'"),
    "gencheck_agree": (0.85, "of those, label agrees with the readout"),
    "state_parsed": (0.8, "state answers that parse to a duration"),
    "mt_steps_parsed": (0.8, "multi-turn step replies with a parsed time horizon"),
}


def main():
    run = Path(sys.argv[1])
    got = {}
    if (run / "choice.parquet").exists():
        c = pd.read_parquet(run / "choice.parquet")
        g = pd.read_parquet(run / "gencheck.parquet")
        ok = g.gen_short.notna()
        got["readout_mass"] = c.mass_ab.median()
        got["gencheck_format"] = ok.mean()
        got["gencheck_agree"] = (g[ok].gen_short == g[ok].readout_short).mean() if ok.any() else 0.0
    if (run / "state.parquet").exists():
        got["state_parsed"] = pd.read_parquet(run / "state.parquet").stated_years.notna().mean()
    if (run / "multiturn.parquet").exists():
        m = pd.read_parquet(run / "multiturn.parquet")
        got["mt_steps_parsed"] = m[m.kind == "step"].h_step_years.notna().mean()
    failed = []
    for k, v in got.items():
        thr, desc = GATES[k]
        status = "ok" if v >= thr else "FAIL"
        print(f"gate {k:16s} {v:6.3f} (>= {thr}) {status}  {desc}")
        if status == "FAIL":
            failed.append(k)
    print("GATES " + ("OK" if not failed else "FAILED: " + ", ".join(failed)))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
