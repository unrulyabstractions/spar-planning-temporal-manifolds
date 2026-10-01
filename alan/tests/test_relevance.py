import numpy as np
from ptm.relevance import nuisance_subspace, project_out, fit_ridge, principal_angle_to_subspace, pair_pull


def test_projecting_out_a_planted_nuisance_restores_the_readout():
    rng = np.random.default_rng(0); n, d = 800, 60
    w = rng.normal(size=d); w /= np.linalg.norm(w)                 # true horizon direction
    v = rng.normal(size=d); v -= v @ w * w; v /= np.linalg.norm(v)  # nuisance direction, orthogonal to w
    t = rng.uniform(-2, 2, n); D = rng.uniform(-2, 2, n)
    X = np.outer(t, w) + 0.3 * rng.normal(size=(n, d))
    Xd = X + np.outer(1.5 * (D - t), v) + 0.3 * rng.normal(size=(n, d))   # displaced along v in proportion to D - t, plus fresh noise
    # (without fresh noise the projected pair difference is pure floating-point residue proportional to D - t, whose rank
    #  correlation with D - t is spuriously high)
    # a ridge fit on a mix of clean and distractor rows leaks v into the readout when v is slightly correlated with the label
    # by construction; here we test the mechanism directly: pairs give v, projection removes the pull
    diffs = Xd - X
    V, energy = nuisance_subspace(diffs, 1)
    assert abs(abs(V[:, 0] @ v) - 1) < 1e-3 and energy[0] > 0.4   # noise spreads the rest of the energy
    r = fit_ridge(X, t)
    # a decoder that has a component along v is pulled; simulate by adding v to the weight
    r.coef_ = r.coef_ + 0.5 * v
    pull_before = pair_pull(r.predict(Xd), r.predict(X), D, t)
    r2 = fit_ridge(project_out(X, V), t); pull_after = pair_pull(r2.predict(project_out(Xd, V)), r2.predict(project_out(X, V)), D, t)
    naive_after = np.corrcoef(r2.predict(project_out(Xd, V)) - t, D - t)[0, 1]   # residual-based: inflated by attenuation
    print(f"pair pull before {pull_before:.3f} after {pull_after:.3f}   (naive residual correlation after: {naive_after:.3f})")
    assert pull_before > 0.5 and abs(pull_after) < 0.1 and naive_after > abs(pull_after)
    assert abs(principal_angle_to_subspace(v, np.column_stack([w]) ) - 90) < 1e-6
    assert principal_angle_to_subspace(w, np.column_stack([w])) < 1e-4
