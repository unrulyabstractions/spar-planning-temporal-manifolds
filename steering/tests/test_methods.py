"""Method bundles (bundle.py, steer/methods.py) on synthetic activations over the real bank (tests/fixtures.py)."""

import json
import random

import numpy as np
import pytest
import torch

import bundle
import fit
from fixtures import BANK, make_activations
from pbank.records import load_prompts
from steer.edits import CurveEdit
from steer.manifold import Manifold
from steer.methods import LINEAR, METHODS, load, relabel


@pytest.fixture
def bundles(tmp_path):
    acts = make_activations(tmp_path / "acts", splits=("extraction", "verification"))
    fit.main(["--activations", str(acts), "--bank", str(BANK), "--layer", "15"], fits_root=tmp_path / "fits")
    fit_dir = tmp_path / "fits" / "acts" / "default" / "L15_label"
    outs = bundle.main(["--fit", str(fit_dir), "--activations", str(acts)], methods_root=tmp_path / "methods")
    return acts, fit_dir, outs, tmp_path


def test_every_method_is_bundled_and_loads_with_its_parameters(bundles):
    acts, fit_dir, outs, _ = bundles
    assert [o.name.split("__")[0] for o in outs] == list(METHODS)
    directions = np.load(fit_dir / "directions.npz")
    manifold = Manifold.load(fit_dir / "manifold.npz")
    for out in outs:
        m = load(out)
        meta = m.meta
        print(f"{out.parent.name}/{out.name}: kind {m.kind}, layer {m.layer}, site {m.site}, positions {m.positions}, "
              f"x = {meta['request']['name']} ({meta['request']['unit'][:40]}...), fitted on {meta['fitted_on']['n_prompts']} "
              f"prompts, include_none {meta['fitted_on']['include_none']}")
        assert (m.layer, m.site, m.positions) == (15, "label", "all")
        assert meta["fitted_on"]["activations_array_sha256"] == json.loads((fit_dir / "fit.json").read_text())["inputs"]["array_sha256"]
        assert sorted(meta["files"]) == ["arrays.npz", "fit.json", "fit_uids.jsonl", "search.jsonl"]
        assert len(m.fit_uids()) == 471
        if m.kind == "push":
            key = LINEAR[meta["method"]][0]             # e.g. caa-near_only is stored as caa_near_only
            assert np.array_equal(m.structure["direction"], directions[key])
            assert m.structure["typical_norm"] == float(directions["typical_norm"])
        else:
            for name in ("mu", "U", "t_grid", "curve"):
                assert np.array_equal(getattr(m.structure["manifold"], name), getattr(manifold, name))
            assert m.edit.mode == meta["method"].split("-")[1]


def test_loaded_edits_act_like_the_numpy_references(bundles):
    acts, fit_dir, outs, _ = bundles
    X = np.load(acts / "L15_label.npy")[:40].astype(np.float64)
    h = torch.tensor(X)[None]
    m = load(next(o for o in outs if o.name.startswith("curve-snap")))
    edit64 = CurveEdit(m.structure["manifold"], m.edit.mode, dtype=torch.float64)
    ref = Manifold.load(fit_dir / "manifold.npz").steer(X, 0.5, "snap")
    err = np.abs(edit64(h, 0.5)[0].numpy() - ref).max()
    assert m.edit.dtype == torch.float32                  # what a loaded bundle runs in by default
    print(f"loaded curve-snap (float64) vs Manifold.steer on the fit's own curve: max |diff| {err:.1e}")
    assert err < 1e-10


def test_a_tampered_bundle_is_refused(bundles):
    _, _, outs, _ = bundles
    with (outs[0] / "fit_uids.jsonl").open("a") as f:
        f.write("\n")
    with pytest.raises(ValueError, match="SHA-256"):
        load(outs[0])


def test_refit_on_the_same_prompts_reproduces_the_method_exactly(bundles):
    acts, _, outs, _ = bundles
    for out in outs:                                  # every method, the within-regime CAAs included
        m = load(out)
        again = m.refit(acts, BANK)
        if m.kind == "push":
            same = np.array_equal(again.structure["direction"], m.structure["direction"])
        else:
            same = all(np.array_equal(getattr(again.structure["manifold"], k), getattr(m.structure["manifold"], k))
                       for k in ("mu", "U", "t_grid", "curve"))
        print(f"refit of {m.meta['method']} on its own {len(m.fit_uids())} prompts: identical parameters {same}")
        assert same


def test_refit_with_shuffled_horizons_is_a_label_shuffled_null(bundles):
    acts, _, outs, _ = bundles
    caa = load(outs[0])
    rows = caa.fit_uids()
    rng = random.Random(0)
    strata = {}
    for r in rows:                                     # shuffle within option order, as Augusto's N4 proposes
        strata.setdefault(r["near_first"], []).append(r)
    horizons = {}
    for group in strata.values():
        values = [r["horizon_grid_value"] for r in group]
        rng.shuffle(values)
        horizons.update({r["uid"]: v for r, v in zip(group, values)})
    null = caa.refit(acts, BANK, horizons=horizons)
    cos = float(null.structure["direction"] @ caa.structure["direction"])
    print(f"CAA refitted on within-order shuffled horizons ({null.meta['refit']['relabelled']} prompts relabelled): "
          f"cosine with the real CAA {cos:+.3f}")
    assert abs(cos) < 0.9 and null.meta["refit_of"] == caa.meta["method_id"]


def test_relabel_reproduces_the_banks_h_over_d():
    recs = [r for r in load_prompts(BANK) if r.horizon]
    same = sum(relabel(r, r.horizon).h_over_far_delay == r.h_over_far_delay for r in recs)
    print(f"relabel(record, its own stated horizon): H/D identical for {same}/{len(recs)} prompts")
    assert same == len(recs)


def test_refit_refuses_prompts_that_are_never_fitted_on(bundles):
    acts, _, outs, _ = bundles
    caa = load(outs[0])
    verification = [r.uid for r in load_prompts(BANK) if r.split == "verification"][:3]
    with pytest.raises(ValueError, match="never fitted on"):
        caa.refit(acts, BANK, uids=[row["uid"] for row in caa.fit_uids()] + verification)


def test_bundle_refuses_overwrite_and_foreign_activations(bundles):
    acts, fit_dir, _, tmp_path = bundles
    with pytest.raises(SystemExit, match="never overwritten"):
        bundle.main(["--fit", str(fit_dir), "--activations", str(acts)], methods_root=tmp_path / "methods")
    other = make_activations(tmp_path / "other", seed=1)
    with pytest.raises(SystemExit, match="was not fitted from"):
        bundle.main(["--fit", str(fit_dir), "--activations", str(other)], methods_root=tmp_path / "m2")


def test_a_fit_without_a_regime_bundles_what_it_has(tmp_path):
    # Without the "none" regime there is no caa-none: the default bundles the rest, and asking for it is refused.
    acts = make_activations(tmp_path / "acts")
    fit.main(["--activations", str(acts), "--bank", str(BANK), "--layer", "15", "--no-include-none", "--name", "nn"],
             fits_root=tmp_path / "fits")
    fit_dir = tmp_path / "fits" / "acts" / "nn" / "L15_label"
    with pytest.raises(SystemExit, match="caa-none"):
        bundle.main(["--fit", str(fit_dir), "--activations", str(acts), "--methods", "caa-none"],
                    methods_root=tmp_path / "m1")
    outs = bundle.main(["--fit", str(fit_dir), "--activations", str(acts)], methods_root=tmp_path / "methods")
    names = [o.name.split("__")[0] for o in outs]
    print(f"bundled from a fit without 'none': {names}")
    assert names == [m for m in METHODS if m != "caa-none"]
    # A fit.json written before the within-regime settings existed bundles with their defaults.
    old = json.loads((fit_dir / "fit.json").read_text())
    assert load(outs[0]).meta["fit_settings"]["within_pool"] == ["near_only", "both"]
    del old["settings"]["within_quantile"], old["settings"]["within_pool"]
    (fit_dir / "fit.json").write_text(json.dumps(old))
    out = bundle.main(["--fit", str(fit_dir), "--activations", str(acts), "--methods", "caa"], methods_root=tmp_path / "m2")
    assert load(out[0]).meta["fit_settings"]["within_quantile"] == 1 / 3
