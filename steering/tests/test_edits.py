"""The torch edits (steer/edits.py) against their numpy references, and the properties the hook relies on.

The curve is the synthetic helix of test_manifold.py (48 log-spaced horizons, 10 prompts each, in R^64).
"""

import numpy as np
import pytest
import torch

from steer.directions import random_directions, steer_linear
from steer.edits import CurveEdit, LinearEdit
from steer.manifold import fit_manifold
from test_manifold import synthetic


@pytest.fixture(scope="module")
def helix():
    X, t, groups, _, _ = synthetic()
    return X, fit_manifold(X, t, groups, k=3, bandwidth=0.3)


def as_batch(X, batch=4):
    """[n, d] -> [batch, n // batch, d] float64 tensor (rows of the batch are different prompts' tokens)."""
    n = (len(X) // batch) * batch
    return torch.tensor(X[:n].astype(np.float64)).reshape(batch, n // batch, -1)


def test_linear_edit_matches_steer_linear_per_row(helix):
    X, _ = helix
    u = random_directions(X.shape[1], 1, seed=3)[0]
    h = as_batch(X)
    x = torch.tensor([0.3, -0.2, 0.0, 1.5], dtype=torch.float64)
    out = LinearEdit(u, typical_norm=7.0, dtype=torch.float64)(h, x)
    ref = np.stack([steer_linear(h[i].numpy(), u, float(x[i]), 7.0) for i in range(4)])
    err = np.abs(out.numpy() - ref).max()
    print(f"linear edit vs steer_linear, per-row strengths {x.tolist()}: max |diff| {err:.2e}")
    assert err < 1e-12


@pytest.mark.parametrize("mode", ["snap", "displace"])
@pytest.mark.parametrize("delta", [-1.0, 0.0, 0.5, 10.0])
def test_curve_edit_matches_manifold_steer(helix, mode, delta):
    X, m = helix
    h = as_batch(X)
    out = CurveEdit(m, mode, dtype=torch.float64)(h, delta).numpy()
    ref = m.steer(h.numpy().reshape(-1, X.shape[1]), delta, mode).reshape(out.shape)
    err = np.abs(out - ref).max()
    print(f"{mode:8s} delta {delta:+5.1f}: torch float64 vs Manifold.steer max |diff| {err:.2e}")
    assert err < 1e-10


def test_curve_edit_in_float32(helix):
    # float32 can pick a neighbouring polyline vertex at near-ties, so compare vertex indices (t-hat values themselves
    # are stored at different precisions) and allow one vertex; where the vertex agrees, the displacement must agree
    # to float32 rounding.
    X, m = helix
    h = as_batch(X)
    e32, e64 = CurveEdit(m, "snap", dtype=torch.float32), CurveEdit(m, "snap", dtype=torch.float64)
    spacing = float(np.diff(m.t_grid).max())
    i32 = torch.round((e32.t_hat(h.float()).double() - m.t_grid[0]) / spacing).long()
    i64 = torch.round((e64.t_hat(h) - m.t_grid[0]) / spacing).long()
    d32, d64 = e32.delta(h.float(), 0.5).double(), e64.delta(h, 0.5)
    rel = (d32 - d64).norm(dim=-1) / d64.norm(dim=-1)
    same = i32 == i64
    print(f"float32 vs float64 (vertex spacing {spacing:.4f} decade): same vertex on {same.float().mean():.1%} of tokens, "
          f"max vertex difference {(i32 - i64).abs().max()}; displacement relative error where the vertex agrees "
          f"max {rel[same].max():.1e}, elsewhere max {rel[~same].max() if (~same).any() else 0:.1e}")
    assert (i32 - i64).abs().max() <= 1 and same.float().mean() > 0.95
    assert rel[same].max() < 1e-4


def test_zero_requests_give_exactly_zero_displacements(helix):
    X, m = helix
    u = random_directions(X.shape[1], 1, seed=3)[0]
    for dtype in (torch.float32, torch.float64):
        h = as_batch(X).to(dtype)
        lin = LinearEdit(u, 7.0, dtype=dtype).delta(h, 0.0)
        disp = CurveEdit(m, "displace", dtype=dtype).delta(h, 0.0)
        snap = CurveEdit(m, "snap", dtype=dtype).delta(h, 0.0)
        print(f"{dtype}: |delta| at x = 0: linear {lin.abs().max():.1e}, displace {disp.abs().max():.1e}, "
              f"snap {snap.norm(dim=-1).max():.3f} (snap removes the off-curve offset)")
        assert torch.equal(lin, torch.zeros_like(lin)) and torch.equal(disp, torch.zeros_like(disp))
        assert snap.norm(dim=-1).max() > 1e-3


def test_per_row_requests_equal_separate_calls(helix):
    X, m = helix
    h = as_batch(X)
    e = CurveEdit(m, "displace", dtype=torch.float64)
    together = e.delta(h, torch.tensor([-1.0, 0.0, 0.25, 2.0], dtype=torch.float64))
    apart = torch.cat([e.delta(h[i:i + 1], s) for i, s in enumerate([-1.0, 0.0, 0.25, 2.0])])
    assert torch.equal(together, apart)
    with pytest.raises(ValueError, match="shape"):
        e.delta(h, torch.tensor([1.0, 2.0]))


def test_curve_edit_chunking_does_not_change_the_result(helix):
    X, m = helix
    h = as_batch(X)
    assert torch.equal(CurveEdit(m, "snap", dtype=torch.float64, chunk=7).delta(h, 0.5),
                       CurveEdit(m, "snap", dtype=torch.float64).delta(h, 0.5))
