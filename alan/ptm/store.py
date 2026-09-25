"""On-disk layout for a capture run.

    run_dir/
      meta.json          model, layers, d_model, position labels, capture + dataset config
      index.parquet      one row per sample: prompt parameters, generation, behavior, shard/row
      acts_0000.safetensors ...   tensor "acts": [n, n_layers+1, n_pos, d_model] bf16

Rows in index.parquet are in the same order as rows across shards, and each row records its
(shard, row) so lookups never rely on ordering alone.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .capture import SampleResult
from .prompts import PromptSample


class RunWriter:
    def __init__(self, run_dir: Path, meta: dict, shard_rows: int = 64):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta = dict(meta)
        self.shard_rows = shard_rows
        self.rows: list[dict] = []
        self._buf: list[torch.Tensor] = []
        self._shard = 0
        self._samples_by_uid: dict[str, PromptSample] = {}

    def register_samples(self, samples: Iterable[PromptSample]) -> None:
        for s in samples:
            self._samples_by_uid[s.sample_uid] = s

    def __call__(self, results: list[SampleResult]) -> None:
        for r in results:
            s = self._samples_by_uid[r.sample_uid]
            row = asdict(s)
            row.update(
                gen_text=r.gen_text,
                gen_ids=json.dumps(r.gen_ids),
                n_generated=r.n_generated,
                transition_tokens=json.dumps(r.transition_tokens),
                response_tokens=json.dumps(r.response_tokens),
                pos_valid=json.dumps(r.pos_valid),
                choice=r.choice,
                choice_gen_index=r.choice_gen_index,
                logit_a=r.logit_a, logit_b=r.logit_b, p_a=r.p_a, p_b=r.p_b,
                shard=self._shard, row=len(self._buf),
            )
            # behavior in short/long terms, independent of label order
            if r.choice is None:
                row["chose_short"] = None
                row["p_short"] = np.nan          # pairwise: sigmoid(logit_short - logit_long), float32
                row["p_short_vocab"] = np.nan    # full-vocab softmax mass on the short label token
                row["p_long_vocab"] = np.nan
            else:
                chose_a = r.choice == "a"
                row["chose_short"] = bool(chose_a == s.short_first)
                la, lb = (r.logit_a, r.logit_b) if s.short_first else (r.logit_b, r.logit_a)
                row["p_short"] = float(1.0 / (1.0 + np.exp(-(la - lb))))
                pa, pb = r.p_a, r.p_b
                row["p_short_vocab"], row["p_long_vocab"] = (pa, pb) if s.short_first else (pb, pa)
            self.rows.append(row)
            self._buf.append(r.acts)
            if len(self._buf) >= self.shard_rows:
                self._flush()

    def _flush(self) -> None:
        if not self._buf:
            return
        acts = torch.stack(self._buf).contiguous()
        save_file({"acts": acts}, str(self.dir / f"acts_{self._shard:04d}.safetensors"))
        self._buf = []
        self._shard += 1
        self._write_index()

    def _write_index(self) -> None:
        pd.DataFrame(self.rows).to_parquet(self.dir / "index.parquet", index=False)
        with open(self.dir / "meta.json", "w") as f:
            json.dump(self.meta, f, indent=2)

    def close(self) -> None:
        self._flush()
        self._write_index()


class RunData:
    """Read side. Activations are sliced lazily from shards, so a 10 GB run opens instantly."""

    def __init__(self, run_dir: Path):
        self.dir = Path(run_dir)
        with open(self.dir / "meta.json") as f:
            self.meta = json.load(f)
        self.index = pd.read_parquet(self.dir / "index.parquet")
        self.labels: list[str] = self.meta["position_labels"]
        self.n_layers: int = self.meta["n_layers"]
        self._shards = sorted(self.dir.glob("acts_*.safetensors"))

    def pos_index(self, label: str) -> int:
        return self.labels.index(label)

    def get(self, layer: int, label: str, rows: pd.DataFrame | None = None) -> np.ndarray:
        """Return float32 [n, d] for one (layer, position), rows in `index` order (or a subset)."""
        df = self.index if rows is None else rows
        p = self.pos_index(label)
        chunks: dict[int, np.ndarray] = {}
        for shard_id in sorted(df["shard"].unique()):
            with safe_open(str(self._shards[shard_id]), framework="pt") as f:
                sl = f.get_slice("acts")
                block = sl[:, layer : layer + 1, p : p + 1, :]           # [n_shard, 1, 1, d]
                chunks[shard_id] = block.reshape(block.shape[0], -1).float().numpy()
        out = np.empty((len(df), chunks[next(iter(chunks))].shape[1]), dtype=np.float32)
        for i, (sh, rw) in enumerate(zip(df["shard"].to_numpy(), df["row"].to_numpy())):
            out[i] = chunks[sh][rw]
        return out

    def get_layer(self, layer: int) -> np.ndarray:
        """Return float32 [n, n_pos, d] for one layer, rows in `index` order. One sequential read per shard."""
        df = self.index
        chunks = {}
        for shard_id in sorted(df["shard"].unique()):
            with safe_open(str(self._shards[shard_id]), framework="pt") as f:
                block = f.get_slice("acts")[:, layer : layer + 1, :, :]      # [n_shard, 1, n_pos, d]
                chunks[shard_id] = block[:, 0].float().numpy()
        n_pos, d = chunks[next(iter(chunks))].shape[1:]
        out = np.empty((len(df), n_pos, d), dtype=np.float32)
        for i, (sh, rw) in enumerate(zip(df["shard"].to_numpy(), df["row"].to_numpy())):
            out[i] = chunks[sh][rw]
        return out

    def valid_mask(self, label: str) -> np.ndarray:
        p = self.pos_index(label)
        return np.array([json.loads(v)[p] for v in self.index["pos_valid"]], dtype=bool)
