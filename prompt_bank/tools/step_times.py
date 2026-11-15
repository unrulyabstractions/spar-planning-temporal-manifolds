"""Generation time per prompt and per decoding step, from a run's own records (no GPU, no model).

    uv run --no-project --with numpy --with pandas --with pyarrow python prompt_bank/tools/step_times.py --pilot
    uv run --no-project --with numpy python prompt_bank/tools/step_times.py --run prompt_bank/runs/<model>__<bank>

A batch decodes until its longest reply stops, so a batch's decoding steps = the largest n_generated in it, and
ms per step = batch seconds / that maximum (prefill is a few steps' worth and is ignored). Seconds per prompt per GPU
= batch seconds / batch size.

--pilot reads the 2026-10-17 Gemma 4 31B thinking pilot, which ran as two independent processes (g0, g1), one per
A100, 192 prompts each at batch 4: per-batch seconds from alan/runs/vast_logs_20261017c/gemma-4-31b_think_pilot_g?.log
and per-reply n_generated from alan/runs/gemma-4-31b_think_pilot/index.parquet (g0's 192 rows, then g1's, in order).
--run reads a prompt_bank baseline run: responses.jsonl rows in file order, batched by run.json's batch_size, where
each row's `seconds` is its batch's wall time divided by the batch size (pbank/local_model.py).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def report(name: str, batch_seconds: list[float], batch_max: list[int], batch_size: int, n_gen: np.ndarray,
           gpus: int = 1, cap: int | None = None) -> None:
    sec, mx = np.array(batch_seconds), np.array(batch_max)
    ms = 1000 * sec / mx
    steps_per_prompt = mx.mean() / batch_size
    print(f"{name}: {len(sec)} batches of {batch_size} on {gpus} GPU(s); replies: n = {len(n_gen)}, "
          f"median {np.median(n_gen):.0f}, mean {n_gen.mean():.0f}, max {n_gen.max()}"
          + (f", at the {cap} cap {int((n_gen >= cap).sum())}" if cap else ""))
    print(f"  total batch time {sec.sum():.0f} s, i.e. {sec.sum() / gpus:.0f} s wall on {gpus} GPU(s)")
    print(f"  seconds per prompt per GPU: {sec.sum() / len(n_gen):.1f}   (wall, all GPUs: {sec.sum() / gpus / len(n_gen):.1f})")
    print(f"  ms per decoding step: median {np.median(ms):.1f} (IQR {np.percentile(ms, 25):.1f}-{np.percentile(ms, 75):.1f}); "
          f"decoding steps per prompt (mean batch maximum / batch size): {steps_per_prompt:.0f}")
    for lo, hi in [(0, 1500), (1500, 3000), (3000, 10 ** 9)]:
        sel = (mx > lo) & (mx <= hi)
        if sel.any():
            print(f"    batches whose longest reply is in ({lo}, {hi if hi < 10 ** 9 else 'inf'}]: {sel.sum():3d}, "
                  f"ms per step median {np.median(ms[sel]):.1f}")
    print(f"  700 prompts at {sec.sum() / len(n_gen):.1f} s per prompt per GPU: {700 * sec.sum() / len(n_gen) / 3600:.1f} GPU-hours")


def pilot() -> None:
    import pandas as pd
    n = pd.read_parquet(REPO / "alan/runs/gemma-4-31b_think_pilot/index.parquet")["n_generated"].to_numpy()
    secs, maxes = [], []
    for proc, offset in (("g0", 0), ("g1", 192)):
        log = (REPO / f"alan/runs/vast_logs_20261017c/gemma-4-31b_think_pilot_{proc}.log").read_text()
        batch = [float(m.group(1)) for m in re.finditer(r"\] batch ([\d.]+)s", log)]
        assert len(batch) == 48, (proc, len(batch))
        secs += batch
        maxes += [int(n[offset + 4 * k: offset + 4 * k + 4].max()) for k in range(48)]
    report("2026-10-17 pilot (alan/ pilot prompts, cap 8192)", secs, maxes, 4, n, gpus=2, cap=8192)


def run(path: Path) -> None:
    meta = json.loads((path / "run.json").read_text())
    bs, cap = meta["settings"]["batch_size"], meta["settings"]["max_new_tokens"]
    rows = [json.loads(line) for line in (path / "responses.jsonl").open()]
    batches = [rows[i:i + bs] for i in range(0, len(rows), bs)]
    secs = [b[0]["seconds"] * len(b) for b in batches]
    maxes = [max(r["n_generated"] for r in b) for b in batches]
    report(f"{path.name} (cap {cap})", secs, maxes, bs, np.array([r["n_generated"] for r in rows]), cap=cap)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pilot", action="store_true", help="the 2026-10-17 Gemma thinking pilot (alan/runs)")
    p.add_argument("--run", type=Path, help="a prompt_bank baseline run directory")
    args = p.parse_args()
    if not (args.pilot or args.run):
        p.error("give --pilot and/or --run")
    if args.pilot:
        pilot()
    if args.run:
        run(args.run)
