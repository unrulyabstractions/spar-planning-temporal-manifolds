"""The on-disk format written by extract.py, and its reader.

    activations/<name>/
        extract.json          settings, model and software, the bank's and run's identities, counts, array hashes
        index.jsonl           one row per extracted prompt, in array-row order (fields below)
        skipped.jsonl         one row per prompt not extracted, with the reason
        L38_label.npy         float32 [n, d]: residual stream at layer 38, at the answer-label token
        L38_prelabel.npy      float32 [n, d]: the same at the token before it

index.jsonl fields: `row` (array row), `uid`, `choice_label` and `chosen_option` (from the baseline run),
`label_token` and `prelabel_token` (the decoded tokens at the two sites, for auditing), `label_index` (position of
the label token in the generated ids), `seq_len` (tokens in the replayed sequence, prompt included),
`replay_label_margin` (at the prelabel site of the replayed pass: logit of the label token the model generated
minus the largest other logit; >= 0 for a faithful replay of a greedy generation, except near-ties, which can come
out one bf16 step, about 0.125, negative) and `replay_predicts_label` (margin >= 0).

extract_horizon.py writes the same layout for the horizon site (the prompt tokens that state the horizon), in its
own directory:

    activations/<model>[-nothink]__<bank>__horizon/
        extract.json          as above; settings.thinking is part of the model input and so of the token indices
        index.jsonl           one row per array row, i.e. per horizon token of a prompt (fields below)
        horizon_sites.jsonl   one row per prompt, including those without a horizon (fields below)
        L00_horizon.npy ...   float32 [rows, d], one file per layer; a prompt has as many rows as its horizon has
                              tokens ("15 minutes": "1", "5", " minutes")

index.jsonl fields there: `row`, `uid`, `k` (which token of the horizon, from 0), `n_tokens` (tokens in this
horizon), `token_index` (the token's index in the model input, chat template included, no padding), `token_id`,
`token` (decoded on its own).

horizon_sites.jsonl fields: `uid`, `split`, `regime`, `horizon` (as written), `horizon_grid_value`, `log10_days`
(of the grid value), `near_first`, `label_style` ("a)/b)" or "1)/2)"), `far_label`, `char_start` / `char_end` (the
horizon in the prompt text), `n_input_tokens`, `token_indices`, `token_ids`, `tokens`, `rows` (its array rows).
Prompts without a horizon have empty lists and nulls. These rows can be passed as the hook's `rows`, and
`token_indices` to `steer.hooks.at_prompt`, to edit only the horizon's tokens.

`load(directory, layer, "horizon")` reads one layer of it; `horizon_sites(directory)` reads the per-prompt rows.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pbank.records import read_jsonl

POSITIONS = ("label", "prelabel")   # the sites extract.py reads, and fit.py can fit on
HORIZON = "horizon"                 # the site extract_horizon.py reads


def array_name(layer: int, position: str) -> str:
    return f"L{layer:02d}_{position}.npy"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ActivationSet:
    """One layer and site of an extraction: X[i] is the activation of prompt index[i]["uid"]."""
    X: np.ndarray          # float32 [n, d]
    index: list[dict]      # index.jsonl rows, in row order
    layer: int
    position: str
    meta: dict             # extract.json

    @property
    def uids(self) -> list[str]:
        return [row["uid"] for row in self.index]


def load(directory: Path, layer: int, position: str) -> ActivationSet:
    """Read one layer and site, checking the array against the hash recorded in extract.json."""
    directory = Path(directory)
    meta = json.loads((directory / "extract.json").read_text())
    name = array_name(layer, position)
    if name not in meta["arrays"]:
        raise FileNotFoundError(f"{directory} has no {name}; extracted: {sorted(meta['arrays'])}")
    if file_sha256(directory / name) != meta["arrays"][name]["sha256"]:
        raise ValueError(f"{directory / name} does not match the SHA-256 recorded in extract.json")
    X = np.load(directory / name)
    index = read_jsonl(directory / "index.jsonl")
    if len(index) != len(X) or any(row["row"] != i for i, row in enumerate(index)):
        raise ValueError(f"{directory}: index.jsonl does not match {name} row for row")
    return ActivationSet(X=X, index=index, layer=layer, position=position, meta=meta)


def horizon_sites(directory: Path) -> dict[str, dict]:
    """horizon_sites.jsonl of an extract_horizon.py output, by uid, checked against index.jsonl."""
    directory = Path(directory)
    sites = {row["uid"]: row for row in read_jsonl(directory / "horizon_sites.jsonl")}
    index = read_jsonl(directory / "index.jsonl")
    for uid, site in sites.items():
        got = [(index[r]["uid"], index[r]["token_index"]) for r in site["rows"]]
        if got != [(uid, k) for k in site["token_indices"]]:
            raise ValueError(f"{directory}: horizon_sites.jsonl and index.jsonl disagree for {uid}")
    return sites
