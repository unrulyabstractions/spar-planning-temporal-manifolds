"""Reading the shared random-direction draws written by evaluation/shared_draws.py.

The evaluation team's script turns one extraction's activations into the random directions both teams use for one
(model, layer, site), so that "random" means the same vectors in every report:

    draws/<model>/L<layer>_<site>/
        draws.npz        unit vectors, [K, d] float32 per family: isotropic (N1), covariance (N2), real_diff (N2')
        manifest.json    model, layer, site, d, K, seeds, and the SHA-256 of the activations file they were drawn from
        uids.txt         the prompts used (extraction split, with a horizon), sorted
        SHA256SUMS       hashes of the files above

`load_shared` checks the files against SHA256SUMS and the manifest against the fit's own inputs before returning
anything, so a fit cannot silently use draws made for another model, layer, site or extraction. It does not regenerate
the draws; `shared_draws.py verify` does that.

One difference in file naming to know about: shared_draws.py reads activations as `L<layer>_<site>.npy`, and extract.py
writes `L<layer, two digits>_<site>.npy`. They are the same name for layers 10 and above.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from steer.activations import file_sha256

FAMILIES = ("isotropic", "covariance", "real_diff")


@dataclass
class SharedDraws:
    families: dict[str, np.ndarray]    # family -> [K, d] float32 unit vectors, in FAMILIES order
    manifest: dict
    source: dict                       # what fit.json records: where the draws came from, and their hashes


def load_shared(directory: Path, model_id: str, layer: int, site: str, d: int, array_sha256: str) -> SharedDraws:
    """The draws in `directory`, refused unless they were made for this model, layer, site and activations array."""
    directory = Path(directory)
    for name in ("SHA256SUMS", "manifest.json", "draws.npz"):
        if not (directory / name).exists():
            raise FileNotFoundError(f"{directory} has no {name}: not a shared_draws.py output directory")
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        if file_sha256(directory / name) != digest:
            raise ValueError(f"{directory / name} does not match its SHA-256 in SHA256SUMS")
    manifest = json.loads((directory / "manifest.json").read_text())
    expected = {"model": model_id, "layer": layer, "site": site, "d": d}
    wrong = {k: (manifest.get(k), v) for k, v in expected.items() if manifest.get(k) != v}
    if wrong:
        raise ValueError(f"{directory} was made for other inputs (manifest value, this fit's value): {wrong}")
    drawn_from = manifest.get("inputs", {}).get("activations_sha256")
    if drawn_from is not None and drawn_from != array_sha256:
        raise ValueError(f"{directory} was drawn from another activations file (SHA-256 {drawn_from[:12]}, "
                         f"this fit reads {array_sha256[:12]})")
    stored = np.load(directory / "draws.npz")
    families = {f: stored[f] for f in FAMILIES if f in stored}
    if "isotropic" not in families:
        raise ValueError(f"{directory}/draws.npz has no isotropic family")
    for f, v in families.items():
        if v.shape != (manifest["k"], d) or not np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-4):
            raise ValueError(f"{directory}: family {f} is not {manifest['k']} unit vectors of dimension {d}")
    source = {"kind": "shared_draws", "directory": f"{directory.parent.name}/{directory.name}",
              "format": manifest["format"], "k": manifest["k"], "families": list(families),
              "manifest_sha256": file_sha256(directory / "manifest.json"),
              "draws_sha256": file_sha256(directory / "draws.npz"),
              "drawn_from_activations_sha256": drawn_from}
    return SharedDraws(families=families, manifest=manifest, source=source)
