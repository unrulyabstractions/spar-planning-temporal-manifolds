"""The steering edits as torch functions of the residual stream, for use inside the model (steer/hooks.py).

An edit maps activations h [batch, tokens, d] and a request x to a displacement delta(h, x, ctx) of the same shape;
the edited stream is h + delta. Edits compute delta at every position, and the hook decides which positions receive
it (prompt, generated, never padding). So an edit never needs to know where it is applied, and a random control can be
matched to an edit's size token by token from delta itself.

    edit = LinearEdit(direction, typical_norm)     kind "push":   x = strength, a fraction of the typical norm
    edit = CurveEdit(manifold, mode="snap")        kind "target": x = shift along the curve, decades of log10(days)

x is a number, or a tensor [batch] with one value per row. Both edits ignore ctx; it is part of the signature so that
edits that need per-token roles or per-row facts (e.g. centroid substitution, which needs each prompt's stated
horizon) fit the same interface.

These are the torch counterparts of the numpy references `steer.directions.steer_linear` and `Manifold.steer`;
tests/test_edits.py checks that they agree. Arithmetic is done in `dtype` (float32 by default; float64 in tests).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from steer.manifold import STEER_MODES, Manifold

ROLE_PAD, ROLE_PROMPT, ROLE_GENERATED = 0, 1, 2


@dataclass
class EditContext:
    """What a hook knows about the tokens of one forward pass.

    role:     [batch, tokens] int, ROLE_PAD / ROLE_PROMPT / ROLE_GENERATED
    position: [batch, tokens] int, index of the token in its row's sequence (left padding included)
    mask:     [batch, tokens] bool, the positions this pass edits
    rows:     per-row facts supplied by the caller (e.g. uid, stated horizon in log10 days, option order), or None
    """
    role: torch.Tensor
    position: torch.Tensor
    mask: torch.Tensor
    rows: Optional[list[dict]] = None


def _per_row(x, batch: int, device, dtype) -> torch.Tensor:
    """x as a [batch, 1, 1] tensor: a number for every row, or one value per row."""
    x = torch.as_tensor(x, dtype=dtype, device=device)
    if x.ndim == 0:
        x = x.expand(batch)
    if x.shape != (batch,):
        raise ValueError(f"x must be a number or have shape ({batch},), not {tuple(x.shape)}")
    return x[:, None, None]


class _Tensors:
    """numpy arrays of an edit, converted once per (device, dtype)."""

    def __init__(self, **arrays: np.ndarray):
        self._arrays = arrays
        self._cache: dict = {}

    def get(self, device, dtype) -> dict:
        key = (str(device), dtype)
        if key not in self._cache:
            self._cache[key] = {k: torch.as_tensor(np.asarray(v), dtype=dtype, device=device)
                                for k, v in self._arrays.items()}
        return self._cache[key]


@dataclass
class LinearEdit:
    """h + x * typical_norm * direction (CAA, the reward control, a random control: any fixed direction)."""
    direction: np.ndarray            # [d] unit vector
    typical_norm: float
    dtype: torch.dtype = torch.float32
    kind: str = field(default="push", init=False)

    def __post_init__(self):
        self._t = _Tensors(direction=self.direction)

    def delta(self, h: torch.Tensor, x, ctx: Optional[EditContext] = None) -> torch.Tensor:
        u = self._t.get(h.device, self.dtype)["direction"]
        return _per_row(x, h.shape[0], h.device, self.dtype) * self.typical_norm * u.expand(*h.shape)

    def __call__(self, h, x, ctx=None):
        return h.to(self.dtype) + self.delta(h, x, ctx)


@dataclass
class CurveEdit:
    """Slide along the manifold curve by x decades (Manifold.steer's two modes; see steer/manifold.py).

    t-hat is read per token: the log-time of the polyline vertex nearest to the token's subspace coordinates.
      snap:     the subspace part becomes the curve point at t-hat + x
      displace: adds c(t-hat + x) - c(t-hat)
    Both clamp t-hat + x to the curve's range and leave the part orthogonal to the subspace unchanged.
    """
    manifold: Manifold
    mode: str
    dtype: torch.dtype = torch.float32
    chunk: int = 4096                # tokens per nearest-vertex search (bounds the [tokens, grid] distance matrix)
    kind: str = field(default="target", init=False)

    def __post_init__(self):
        if self.mode not in STEER_MODES:
            raise ValueError(f"mode must be one of {STEER_MODES}")
        m = self.manifold
        self._t = _Tensors(mu=m.mu, U=m.U, t_grid=m.t_grid, curve=m.curve)

    def _nearest(self, z: torch.Tensor, curve: torch.Tensor) -> torch.Tensor:
        flat = z.reshape(-1, z.shape[-1])
        c2 = (curve * curve).sum(1)
        idx = torch.cat([((q * q).sum(1, keepdim=True) - 2 * q @ curve.T + c2[None, :]).argmin(1)
                         for q in flat.split(self.chunk)])
        return idx.reshape(z.shape[:-1])

    @staticmethod
    def _point(t: torch.Tensor, t_grid: torch.Tensor, curve: torch.Tensor) -> torch.Tensor:
        """Curve coordinates at log-times t (any shape), clamped to the range, linear between vertices. At a vertex
        the result is that vertex exactly (the interpolation weight is exactly 0 there)."""
        t = t.clamp(t_grid[0], t_grid[-1])
        i = (torch.searchsorted(t_grid, t.contiguous(), right=True) - 1).clamp(0, len(t_grid) - 2)
        frac = ((t - t_grid[i]) / (t_grid[i + 1] - t_grid[i]))[..., None]
        return curve[i] * (1 - frac) + curve[i + 1] * frac

    def t_hat(self, h: torch.Tensor) -> torch.Tensor:
        p = self._t.get(h.device, self.dtype)
        z = (h.to(self.dtype) - p["mu"]) @ p["U"]
        return p["t_grid"][self._nearest(z, p["curve"])]

    def delta(self, h: torch.Tensor, x, ctx: Optional[EditContext] = None) -> torch.Tensor:
        p = self._t.get(h.device, self.dtype)
        z = (h.to(self.dtype) - p["mu"]) @ p["U"]                          # [batch, tokens, k]
        t_hat = p["t_grid"][self._nearest(z, p["curve"])]                 # [batch, tokens]
        shift = _per_row(x, h.shape[0], h.device, self.dtype)[..., 0]     # [batch, 1]
        target = self._point(t_hat + shift, p["t_grid"], p["curve"])
        start = z if self.mode == "snap" else self._point(t_hat, p["t_grid"], p["curve"])
        return (target - start) @ p["U"].T

    def __call__(self, h, x, ctx=None):
        return h.to(self.dtype) + self.delta(h, x, ctx)
