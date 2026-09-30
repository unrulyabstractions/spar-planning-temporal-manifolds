"""Generate the planning conversations and capture activations.

    python scripts/capture.py runs/<name> --model Qwen/Qwen3-14B --batch 16
    python scripts/capture.py runs/pilot --model Qwen/Qwen3-1.7B --smoke     # small local slice
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mtp.prompts import HORIZONS, WORDINGS, build_specs  # noqa: E402
from mtp.run import run  # noqa: E402


def smoke_specs(specs):
    """3 scenarios x 3 horizons x 1 wording (greedy) + 2 no-horizon samples per scenario."""
    sc = ["bakery", "company_emissions", "piano"]
    hs = ["1 week", "1 year", "25 years"]
    w = next(iter(WORDINGS))
    keep = [s for s in specs if s.scenario in sc and (
        (s.condition == "horizon" and s.h_target_text in hs and s.wording == w) or
        (s.condition == "none" and s.sample < 2))]
    return keep


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--model", default="Qwen/Qwen3-14B")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--n-none", type=int, default=16, help="sampled plans per scenario without a horizon")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--gpu-gib", type=float, default=None, help="cap GPU memory, offload the rest (local tests)")
    a = p.parse_args()
    specs = build_specs(a.n_none)
    if a.smoke:
        specs = smoke_specs(specs)
    run(specs, Path(a.out), a.model, batch_size=a.batch, gpu_gib=a.gpu_gib, log=lambda m: print(m, flush=True))
