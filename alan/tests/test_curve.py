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
