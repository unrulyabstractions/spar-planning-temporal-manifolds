import numpy as np
from ptm.curve import Basis, CurveModel


def make_data(n=600, d=40, seed=0, shape="quad", noise=0.3, ctx=False):
    """Planted curve with FIXED directions/offsets (seed 123); `seed` only changes the sample."""
    fixed = np.random.default_rng(123)
    u1, u2 = fixed.normal(size=d), fixed.normal(size=d); u1 /= np.linalg.norm(u1); u2 -= u2 @ u1 * u1; u2 /= np.linalg.norm(u2)
    off = fixed.normal(size=(3, d)) * 2
    rng = np.random.default_rng(seed)
    t = rng.uniform(-2.5, 2.0, n)
    f = 3 * t[:, None] * u1 + (2 * (t ** 2)[:, None] * u2 if shape == "quad" else 0)
    c = None
    X = 5 + f
    if ctx:
        c = rng.choice(["a", "b", "c"], n)
        X = X + off[np.searchsorted(["a", "b", "c"], c)]
    X = X + noise * rng.normal(size=(n, d))
    return X, t, c


def test_forward_fit_recovers_planted_quadratic():
    X, t, _ = make_data(noise=0.1)          # noise ceiling on R2 ≈ 0.985
    for kind, ok in [("line", False), ("quad", True), ("spline", True)]:
        m = CurveModel(Basis(kind), alpha=1e-3).fit(X, t)
        Xt, tt, _ = make_data(n=300, seed=2, noise=0.1)
        r2 = 1 - ((m.predict(tt) - Xt) ** 2).sum() / ((Xt - Xt.mean(0)) ** 2).sum()
        print(f"{kind}: held-out forward R2 {r2:.3f}")
        assert (r2 > 0.9) == ok


def test_decoder_inverts_the_curve():
    X, t, _ = make_data(noise=0.2)
    m = CurveModel(Basis("quad"), alpha=1e-3).fit(X, t)
    Xt, tt, _ = make_data(n=200, seed=3, noise=0.2)
    th, res = m.decode(Xt)
    err = np.abs(th - tt); print(f"decode: median |err| {np.median(err):.3f}, 90th {np.quantile(err, 0.9):.3f}, median residual {np.median(res):.2f}")
    assert np.median(err) < 0.05 and np.quantile(err, 0.9) < 0.15


def test_context_offsets_are_recovered():
    X, t, c = make_data(ctx=True, noise=0.2)
    m0 = CurveModel(Basis("quad"), context="none", alpha=1e-3).fit(X, t)
    m1 = CurveModel(Basis("quad"), context="offset", alpha=1e-3).fit(X, t, c)
    Xt, tt, ct = make_data(n=300, seed=4, ctx=True, noise=0.2)
    r2 = lambda m, cc: 1 - ((m.predict(tt, cc) - Xt) ** 2).sum() / ((Xt - Xt.mean(0)) ** 2).sum()
    print(f"no-context R2 {r2(m0, None):.3f}  offset R2 {r2(m1, ct):.3f}")
    assert r2(m1, ct) > 0.9 > r2(m0, None)
    th, _ = m1.decode(Xt, ct); assert np.median(np.abs(th - tt)) < 0.06


def test_geometry_of_a_planar_parabola():
    X, t, _ = make_data(noise=0.05)
    m = CurveModel(Basis("quad"), alpha=1e-3).fit(X, t)
    g = m.geometry(-2.0, 1.5)
    print(f"tortuosity {g['tortuosity']:.3f} turn {g['turn_deg']:.0f} deg  curve var top {np.round(g['curve_var_top'][:3], 3)}  n95 {g['n95']}  kappa max at t={g['t_kappa_max']:.2f}")
    assert g["n95"] == 2 and g["curve_var_top"][2] < 0.01          # planar
    assert abs(g["t_kappa_max"]) < 0.3                               # parabola 3t u1 + 2t^2 u2: curvature peaks at the vertex t=0
    lin = CurveModel(Basis("line"), alpha=1e-3).fit(X, t).geometry(-2.0, 1.5)
    assert lin["tortuosity"] < 1.001 and lin["n95"] == 1


def test_quantile_knots_stable_on_dense_integer_grid():
    """log10 of the dense integer grid (1..36525) is sparse at the low end. Planted curve + noise with known signal share:
    held-out R² with quantile knots must recover it; equally spaced knots are reported for contrast (the 2026-10-01 bug)."""
    from ptm.curve import cv_r2, quantile_basis
    from ptm.prompts import dense_integer_grid
    v = np.array(dense_integer_grid(1500, 36525)); t = np.log10(v)
    rng = np.random.default_rng(0); d = 60
    U = np.linalg.qr(rng.standard_normal((d, 3)))[0]
    F = np.column_stack([t, np.sin(1.5 * t), 0.3 * t ** 2]) @ U.T
    F -= F.mean(0); noise = rng.standard_normal((len(t), d))
    noise *= np.sqrt((F ** 2).sum() / (noise ** 2).sum())                 # signal share 0.5
    X = F + noise
    r2_q, err, err0 = cv_r2(X, t, lambda: quantile_basis(t, 8))
    r2_u, err_u, _ = cv_r2(X, t, lambda: Basis("spline", t_min=t.min(), t_max=t.max(), n_interior=8))
    low = v < 10
    print(f"\n  planted signal share 0.50: quantile knots R² {r2_q:.3f} (values<10: {1 - err[low].sum() / err0[low].sum():.2f}); "
          f"equally spaced knots R² {r2_u:.3f} (values<10: {1 - err_u[low].sum() / err0[low].sum():.2f})")
    assert abs(r2_q - 0.5) < 0.03
    assert r2_u < 0.3                     # the test must actually exercise the failure it guards against


def test_cv_r2_with_group_offsets():
    """Planted shared curve + per-group offsets: the shared-curve fit misses the offsets, the offset fit recovers them,
    per-group curves add nothing."""
    from ptm.curve import cv_r2, quantile_basis
    rng = np.random.default_rng(1); n, d = 900, 30
    t = rng.uniform(-2, 2, n); g = rng.choice(["days", "weeks", "months"], n)
    U = np.linalg.qr(rng.standard_normal((d, 2)))[0]
    F = np.column_stack([t, np.sin(t)]) @ U.T
    off = {k: rng.standard_normal(d) * 1.0 for k in ["days", "weeks", "months"]}
    X = F + np.array([off[k] for k in g]) + 0.3 * rng.standard_normal((n, d))
    b = lambda: quantile_basis(t, 6)
    r_none, r_off, r_int = (cv_r2(X, t, b, ctx=g, context=c)[0] for c in ("none", "offset", "interaction"))
    print(f"\n  shared {r_none:.3f}  offsets {r_off:.3f}  per-group curves {r_int:.3f}")
    assert r_off > r_none + 0.2 and abs(r_int - r_off) < 0.02


def test_cv_r2_interpolate_only_and_separate_levels():
    """Shared curve + per-level offset truth, levels with different t ranges (years 1..100 on a log grid, months spread).
    Without an outlier, separate per-level curves score like the offset model. With an outlying point at the low end of
    one level, plain K-fold extrapolates to it and collapses; scoring interpolation only excludes it (and only the
    levels' extremes) and recovers most of the score. The rest of the gap is that point's least-squares influence when
    it sits in a training fold, which scoring cannot remove."""
    from ptm.curve import cv_r2, quantile_basis
    rng = np.random.default_rng(2); d = 30
    t = np.concatenate([np.log10(np.arange(1, 101)), rng.uniform(-1, 2, 400)])
    g = np.array(["years"] * 100 + ["months"] * 400)
    U = np.linalg.qr(rng.standard_normal((d, 2)))[0]
    X = np.column_stack([t, np.sin(2 * t)]) @ U.T + np.where(g[:, None] == "years", 1.0, -1.0) * U[:, 0]
    N = 0.1 * rng.standard_normal(X.shape)
    X += N
    ceiling = 1 - (N ** 2).sum() / ((X - X.mean(0)) ** 2).sum()            # R² of the true curve + offsets
    b = lambda: quantile_basis(t, 6)
    lvl = lambda tl: quantile_basis(tl, 4)
    clean_off = cv_r2(X, t, b, ctx=g, context="offset", interpolate_only=True)[0]
    clean_sep = cv_r2(X, t, b, ctx=g, context="separate", interpolate_only=True, make_level_basis=lvl)[0]
    Xo = X.copy(); Xo[0] += 8 * rng.standard_normal(d)                   # "1 year": far from everything
    r_all = cv_r2(Xo, t, b, ctx=g, context="offset")[0]
    r_int, e, _ = cv_r2(Xo, t, b, ctx=g, context="offset", interpolate_only=True)
    print(f"\n  noise ceiling {ceiling:.3f}; clean: offset {clean_off:.3f}, separate {clean_sep:.3f} | outlier: offset all rows {r_all:.3f}, "
          f"interpolation only {r_int:.3f} ({int(np.isnan(e).sum())} excluded)")
    assert clean_off > ceiling - 0.01 and abs(clean_sep - clean_off) < 0.01
    assert np.isnan(e[0]) and int(np.isnan(e).sum()) <= 4 and r_int > r_all + 0.3
    assert cv_r2(X, t, b)[0] == cv_r2(X, t, b, interpolate_only=False)[0]
