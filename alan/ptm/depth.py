"""Layer specifications by absolute index or fractional depth.

A layer spec is "22" (absolute stored-layer index) or "0.55L" (fraction of the model's decoder depth,
resolved as round(f × n_layers) against the run being analysed). Fractional specs make the same analysis
chain apply to models of different depth; the prior work found fractional depth is the coordinate that
compares across models. For Qwen3-14B (40 layers) the standard fractions resolve to the layer indices used
throughout the logs: 0.35L→14, 0.45L→18, 0.55L→22, 0.65L→26, 0.72L→29, 0.825L→33, 0.92L→37.
A cell spec is "<layer spec>:<position>", e.g. "0.55L:T3"; lists are comma-separated.
"""

from __future__ import annotations

import json
from pathlib import Path


def n_layers_of(run) -> int:
    """Accept a RunData, a run directory, or an int."""
    if isinstance(run, int):
        return run
    if hasattr(run, "meta"):
        return int(run.meta["n_layers"])
    return int(json.load(open(Path(run) / "meta.json"))["n_layers"])


def resolve_layer(spec: str | int, run) -> int:
    s = str(spec).strip()
    if s.endswith("L"):
        f = float(s[:-1])
        if not 0.0 <= f <= 1.0:
            raise ValueError(f"fractional depth must be in [0, 1]: {spec}")
        return int(round(f * n_layers_of(run)))
    return int(s)


def resolve_cell(cell: str, run) -> tuple[int, str]:
    layer, pos = cell.split(":")
    return resolve_layer(layer, run), pos


def parse_cells(spec: str, run) -> list[tuple[int, str]]:
    return [resolve_cell(c, run) for c in spec.split(",") if c]


def parse_layers(spec: str, run) -> list[int]:
    return [resolve_layer(x, run) for x in spec.split(",") if x]
