"""Compact activation subsets for sharing: selected layers × positions, one safetensors per layer,
row-chunked so no file exceeds a size cap (GitHub blocks files over 100 MB).

Layout inside a run directory:
    subset.json                        {"layers": [...], "positions": [...], "chunks": [[start, stop], ...], "d_model": d}
    acts_subset_L{layer:02d}_c{k}.safetensors   tensor "acts": [rows_in_chunk, n_pos_sel, d] bf16, rows in index.parquet order
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .store import RunData


def export_subset(run: RunData, layers: list[int], positions: list[str], max_mb: float = 95.0) -> dict:
    d = run.meta["d_model"]; n = len(run.index)
    bytes_per_row = len(positions) * d * 2
    rows_per_chunk = max(1, int(max_mb * 1024 * 1024 // bytes_per_row))
    chunks = [[s, min(s + rows_per_chunk, n)] for s in range(0, n, rows_per_chunk)]
    pos_idx = [run.pos_index(p) for p in positions]
    for l in layers:
        XL = run.get_layer(l)                                         # float32 [n, n_pos, d], exact from bf16
        sel = torch.from_numpy(XL[:, pos_idx, :]).to(torch.bfloat16)  # back to bf16: exact (values were bf16)
        for k, (s, e) in enumerate(chunks):
            save_file({"acts": sel[s:e].contiguous()}, str(run.dir / f"acts_subset_L{l:02d}_c{k}.safetensors"))
    meta = dict(layers=layers, positions=positions, chunks=chunks, d_model=d, n=n,
                layer_convention=run.meta.get("layer_convention"), source_run=run.meta.get("model_name"))
    json.dump(meta, open(run.dir / "subset.json", "w"), indent=2)
    return meta


def load_subset(run_dir: Path | str, layer: int, position: str) -> np.ndarray:
    """float32 [n, d] in index.parquet order, from the subset files (no full shards needed)."""
    run_dir = Path(run_dir); meta = json.load(open(run_dir / "subset.json"))
    if layer not in meta["layers"] or position not in meta["positions"]:
        raise KeyError(f"subset holds layers {meta['layers']} × positions {meta['positions']}")
    p = meta["positions"].index(position)
    out = np.empty((meta["n"], meta["d_model"]), dtype=np.float32)
    for k, (s, e) in enumerate(meta["chunks"]):
        with safe_open(str(run_dir / f"acts_subset_L{layer:02d}_c{k}.safetensors"), framework="pt") as f:
            out[s:e] = f.get_slice("acts")[:, p : p + 1, :][:, 0].float().numpy()
    return out
