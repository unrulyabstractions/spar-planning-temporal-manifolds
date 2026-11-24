"""fit.py end to end on synthetic activations over the real bank (tests/fixtures.py), and the fit-tree comparer."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import fit
from fixtures import BANK, make_activations

STEERING = Path(__file__).resolve().parents[1]


def run_fit(acts, fits_root, *extra):
    fit.main(["--activations", str(acts), "--bank", str(BANK), "--layer", "15", *extra], fits_root=fits_root)
    return fits_root / acts.name / (extra[extra.index("--name") + 1] if "--name" in extra else "default") / "L15_label"


def test_fit_writes_its_outputs_and_recovers_the_planted_structure(tmp_path):
    acts = make_activations(tmp_path / "acts")
    out = run_fit(acts, tmp_path / "fits")
    assert sorted(p.name for p in out.iterdir()) == ["directions.npz", "fit.json", "fit_uids.jsonl", "manifold.npz",
                                                    "per_horizon.jsonl"]
    meta = json.loads((out / "fit.json").read_text())
    d = meta["diagnostics"]
    print(f"synthetic fit: n {meta['n_prompts']}, regimes {meta['regimes']}; cos(CAA, choice) "
          f"{d['caa']['cos_choice']:+.3f}; curve explained {d['manifold']['explained_variance_of_means']}; "
          f"held-out Spearman {d['manifold']['held_out']['spearman']:+.3f}, median error "
          f"{d['manifold']['held_out']['median_abs_err_decades']:.3f} decades")
    assert meta["n_prompts"] == 471 and meta["regimes"] == {"both": 86, "none": 279, "near_only": 106}
    assert d["manifold"]["held_out"]["spearman"] > 0.99
    # Measured 0.136 decades. Most of it is the planted choice offset: the share of far choices rises with the
    # horizon, so the choice direction enters the per-horizon means and their PCs, and a prompt that chose against
    # its horizon's majority reads off-curve (largest for "none", where the fixture alternates choices). With
    # PLANT_CHOICE = 0 the same fit gives 0.047.
    assert d["manifold"]["held_out"]["median_abs_err_decades"] < 0.2


def test_fit_refuses_to_overwrite(tmp_path):
    acts = make_activations(tmp_path / "acts")
    run_fit(acts, tmp_path / "fits")
    with pytest.raises(SystemExit, match="exists"):
        run_fit(acts, tmp_path / "fits")


def test_fit_refuses_activations_from_another_bank(tmp_path):
    acts = make_activations(tmp_path / "acts")
    meta = json.loads((acts / "extract.json").read_text())
    meta["settings"]["bank_prompts_sha256"] = "0" * 64
    (acts / "extract.json").write_text(json.dumps(meta))
    with pytest.raises(SystemExit, match="different bank"):
        run_fit(acts, tmp_path / "fits")


def test_compare_fits_finds_identity_and_one_ulp_changes(tmp_path):
    acts = make_activations(tmp_path / "acts")
    a = run_fit(acts, tmp_path / "A")
    b = run_fit(acts, tmp_path / "B")
    cmd = [sys.executable, str(STEERING / "tools" / "compare_fits.py"), str(a), str(b)]
    same = subprocess.run(cmd, capture_output=True, text=True)
    print(same.stdout.strip().splitlines()[-1])
    assert same.returncode == 0, same.stdout
    z = dict(np.load(b / "directions.npz"))
    z["caa"] = z["caa"].copy()
    z["caa"][0] = np.nextafter(z["caa"][0], np.inf)
    np.savez(b / "directions.npz", **z)
    changed = subprocess.run(cmd, capture_output=True, text=True)
    print([line for line in changed.stdout.splitlines() if line.startswith("DIFFER")])
    assert changed.returncode == 1 and "caa: values differ" in changed.stdout


def fit_json(out):
    return json.loads((out / "fit.json").read_text())


def test_include_none_is_the_default_and_can_be_switched_off(tmp_path):
    acts = make_activations(tmp_path / "acts")
    on = fit_json(run_fit(acts, tmp_path / "fits"))
    off = fit_json(run_fit(acts, tmp_path / "fits", "--no-include-none", "--name", "no-none"))
    print(f"include none: n {on['n_prompts']}, excluded {on['excluded']}, CAA short {on['diagnostics']['caa']['short_by_regime']}")
    print(f"no none:      n {off['n_prompts']}, excluded {off['excluded']}, CAA short {off['diagnostics']['caa']['short_by_regime']}, "
          f"curve range {off['diagnostics']['manifold']['t_range_log10_days']}")
    assert on["settings"]["include_none"] is True and off["settings"]["include_none"] is False
    assert on["excluded"] == {"not_extraction_split": 0, "no_horizon": 29, "none_regime": 0, "excluded_horizon": 0}
    assert off["excluded"]["none_regime"] == 279 and off["n_prompts"] == 192 and "none" not in off["regimes"]
    assert off["diagnostics"]["caa"]["short_by_regime"] == {"near_only": 71}
    assert off["diagnostics"]["manifold"]["t_range_log10_days"][0] == pytest.approx(1.0)   # 10 days


def test_verification_prompts_are_never_fitted_on(tmp_path):
    acts = make_activations(tmp_path / "acts", splits=("extraction", "verification"))
    meta = fit_json(run_fit(acts, tmp_path / "fits"))
    print(f"activations of all 700 prompts: fitted {meta['n_prompts']}, excluded {meta['excluded']}")
    assert meta["excluded"]["not_extraction_split"] == 200 and meta["n_prompts"] == 471


def test_exclude_horizons_leaves_out_exactly_those_prompts(tmp_path):
    acts = make_activations(tmp_path / "acts")
    meta = fit_json(run_fit(acts, tmp_path / "fits", "--exclude-horizons", "3 years", "10 days", "--name", "held-out"))
    lines = (tmp_path / "fits" / "acts" / "held-out" / "L15_label" / "per_horizon.jsonl").read_text().splitlines()
    per_horizon = [json.loads(line) for line in lines]
    from pbank.records import load_prompts
    expected = sum(r.split == "extraction" and r.horizon_grid_value in ("3 years", "10 days") for r in load_prompts(BANK))
    print(f"excluded {meta['excluded']} (bank has {expected} extraction prompts at those horizons); "
          f"{len(per_horizon)} horizons left on the curve")
    assert meta["excluded"]["excluded_horizon"] == expected > 0
    assert {"3 years", "10 days"}.isdisjoint(h["horizon"] for h in per_horizon) and len(per_horizon) == 46
    with pytest.raises(SystemExit, match="not a stated horizon"):
        run_fit(acts, tmp_path / "fits", "--exclude-horizons", "3 yrs", "--name", "typo")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_fit_uids_lists_every_fitted_prompt_with_its_classes(tmp_path):
    acts = make_activations(tmp_path / "acts", splits=("extraction", "verification"))
    out = run_fit(acts, tmp_path / "fits")
    meta, rows = fit_json(out), read_jsonl(out / "fit_uids.jsonl")
    from collections import Counter
    from pbank.records import load_prompts
    bank = {r.uid: r for r in load_prompts(BANK)}
    index = read_jsonl(acts / "index.jsonl")
    caa = Counter(r["caa_class"] for r in rows)
    print(f"fit_uids: {len(rows)} rows; CAA classes {dict(caa)}; reward classes {dict(Counter(r['reward_class'] for r in rows))}; "
          f"folds {dict(sorted(Counter(r['fold'] for r in rows).items()))}")
    print(f"first row: {rows[0]}")
    assert len(rows) == meta["n_prompts"] == 471
    assert caa["long"] == meta["diagnostics"]["caa"]["n_long"] and caa["short"] == meta["diagnostics"]["caa"]["n_short"]
    assert {r["fold"] for r in rows} == set(range(5))
    for row in rows:
        rec = bank[row["uid"]]
        assert index[row["row"]]["uid"] == row["uid"] and rec.split == "extraction"
        assert row["far_label"] == (rec.labels[1] if rec.near_first else rec.labels[0])
        assert (row["caa_class"] == "long") == (rec.h_over_far_delay >= 4)
        assert (row["caa_class"] == "short") == (rec.h_over_far_delay <= 0.25)


def test_search_log_records_every_run_and_only_grows(tmp_path):
    acts = make_activations(tmp_path / "acts")
    fits = tmp_path / "fits"
    run_fit(acts, fits)                                                   # written
    with pytest.raises(SystemExit):
        run_fit(acts, fits)                                               # refused: exists
    first_two = (fits / "acts" / "search.jsonl").read_text()
    run_fit(acts, fits, "--no-include-none", "--name", "no-none")         # written
    with pytest.raises(FileNotFoundError):
        fit.main(["--activations", str(acts), "--bank", str(BANK), "--layer", "12"], fits_root=fits)   # error: no L12
    log = read_jsonl(fits / "acts" / "search.jsonl")
    for row in log:
        print(f"{row['outcome']:8s} {row['settings']['name']:8s} L{row['settings']['layer']} "
              f"include_none={row['settings']['include_none']} {row.get('reason', row.get('out'))}")
    assert [r["outcome"] for r in log] == ["written", "refused", "written", "error"]
    assert (fits / "acts" / "search.jsonl").read_text().startswith(first_two)    # earlier lines untouched
    assert log[0]["headline"]["curve_held_out_spearman"] > 0.99 and log[2]["settings"]["include_none"] is False
    assert all(len(r["code_commit"]) >= 40 or r["code_commit"] == "unknown" for r in log)


# ---- shared random draws (evaluation/shared_draws.py) as the fit's random baseline ---------------------------------

def make_draws(acts, out, layer=15, site="label"):
    """Run the evaluation team's script on the synthetic activations; returns the draws directory."""
    script = STEERING.parent / "evaluation" / "shared_draws.py"
    subprocess.run([sys.executable, str(script), "make", "--activations", str(acts), "--bank",
                    str(BANK / "prompts.jsonl"), "--model", "synthetic/tiny", "--layer", str(layer), "--site", site,
                    "--out", str(out)], check=True, capture_output=True)
    return out / "tiny" / f"L{layer}_{site}"


def test_fit_with_shared_draws_changes_only_the_random_baseline(tmp_path):
    acts = make_activations(tmp_path / "acts")
    draws = make_draws(acts, tmp_path / "draws")
    own = run_fit(acts, tmp_path / "fits")
    shared = run_fit(acts, tmp_path / "fits", "--name", "shared", "--draws", str(draws))
    a, b = np.load(own / "directions.npz"), np.load(shared / "directions.npz")
    stored = np.load(draws / "draws.npz")
    meta = json.loads((shared / "fit.json").read_text())
    src, cos = meta["random_source"], meta["diagnostics"]["caa"]["abs_cos_random"]
    print(f"random_source {src['kind']} {src['directory']}, K {src['k']}, families {src['families']}; "
          f"|cos(CAA, random)| max/p95: " + ", ".join(f"{f} {v['max']:.3f}/{v['p95']:.3f}" for f, v in cos.items()))
    print(f"own fit: {json.loads((own / 'fit.json').read_text())['random_source']}")
    # The methods are untouched; only the random vectors differ, and they are the shared file's, bit for bit.
    for key in ("caa", "reward", "choice", "typical_norm"):
        assert np.array_equal(a[key], b[key]), key
    assert (own / "manifold.npz").read_bytes() == (shared / "manifold.npz").read_bytes()
    assert np.array_equal(b["random"], stored["isotropic"]) and b["random"].shape == (99, 24)
    assert a["random"].shape == (100, 24)
    assert src["families"] == ["isotropic", "covariance", "real_diff"] and src["k"] == 99
    assert src["drawn_from_activations_sha256"] == meta["inputs"]["array_sha256"]
    for f in src["families"]:
        expect = np.abs(stored[f] @ b["caa"]).max()
        assert cos[f]["n"] == 99 and abs(cos[f]["max"] - expect) < 1e-6
    assert meta["diagnostics"]["caa"]["max_abs_cos_random"] == cos["isotropic"]["max"]
    assert "draws" not in meta["settings"]


def test_fit_refuses_draws_made_for_other_inputs(tmp_path):
    acts = make_activations(tmp_path / "acts", layers=(12, 15))
    other = make_activations(tmp_path / "other", seed=1)
    cases = {"another site": make_draws(acts, tmp_path / "d1", site="prelabel"),
             "another layer": make_draws(acts, tmp_path / "d2", layer=12),
             "another extraction": make_draws(other, tmp_path / "d3")}
    tampered = make_draws(acts, tmp_path / "d4")
    (tampered / "uids.txt").write_text("x0001\n")
    cases["a file changed after SHA256SUMS"] = tampered
    for n, (what, draws) in enumerate(cases.items()):
        with pytest.raises(SystemExit) as e:
            run_fit(acts, tmp_path / "fits", "--name", f"r{n}", "--draws", str(draws))
        print(f"{what}: refused: {str(e.value.code)[:110]}")
        assert str(e.value.code).startswith("--draws:")
    log = [json.loads(line) for line in (tmp_path / "fits" / "acts" / "search.jsonl").read_text().splitlines()]
    assert [row["outcome"] for row in log] == ["refused"] * 4


# ---- within-regime CAA -----------------------------------------------------------------------------------------------

def test_within_regime_caa_classes_and_directions(tmp_path):
    from pbank.records import load_prompts, read_jsonl
    acts = make_activations(tmp_path / "acts")
    out = run_fit(acts, tmp_path / "fits")
    w = json.loads((out / "fit.json").read_text())["diagnostics"]["caa_within"]
    d = np.load(out / "directions.npz")
    rows = read_jsonl(out / "fit_uids.jsonl")
    recs = {r.uid: r for r in load_prompts(BANK)}
    X = np.load(acts / "L15_label.npy").astype(np.float64)
    for g, v in w["by_regime"].items():
        print(f"within {g}: {v['n_long']} long (t >= {v['long_min_log10_days']:.2f}) / {v['n_short']} short "
              f"(t <= {v['short_max_log10_days']:.2f}) of {v['n_regime']}; cos with choice {v['cos_choice']:+.3f}, "
              f"with CAA {v['cos_caa']:+.3f}")
    print(f"pooled over {w['pool']}: cos with choice {w['pooled']['cos_choice']:+.3f}; agreement {w['agreement']}; "
          f"CAA's own cos with choice {json.loads((out / 'fit.json').read_text())['diagnostics']['caa']['cos_choice']:+.3f}")
    # The classes on the real bank, by log-horizon thirds within each regime.
    sizes = {g: (v["n_long"], v["n_short"]) for g, v in w["by_regime"].items()}
    assert sizes == {"none": (94, 97), "near_only": (43, 36), "both": (30, 36)}
    assert w["pool"] == ["near_only", "both"] and w["not_formed"] == [] and w["quantile"] == 1 / 3
    # Every prompt's class is within its own regime, and the directions are the stated differences of means.
    from steer.fitting import log10_days
    raw = {}
    for g, v in w["by_regime"].items():
        long_ = [r["row"] for r in rows if r["regime"] == g and r["caa_within_class"] == "long"]
        short = [r["row"] for r in rows if r["regime"] == g and r["caa_within_class"] == "short"]
        assert (len(long_), len(short)) == sizes[g]
        t_long = [log10_days(recs[r["uid"]].horizon_grid_value) for r in rows if r["row"] in set(long_)]
        t_short = [log10_days(recs[r["uid"]].horizon_grid_value) for r in rows if r["row"] in set(short)]
        t_mid = [log10_days(r["horizon_grid_value"]) for r in rows if r["regime"] == g and r["caa_within_class"] is None]
        assert max(t_short) < min(t_mid) and max(t_mid) < min(t_long)
        raw[g] = X[long_].mean(0) - X[short].mean(0)
        assert np.allclose(d[f"caa_{g}"], raw[g] / np.linalg.norm(raw[g]), atol=1e-12)
        assert abs(float(d[f"caa_{g}_raw_norm"]) - np.linalg.norm(raw[g])) < 1e-12
    n = {g: w["by_regime"][g]["n_regime"] for g in ("near_only", "both")}
    pooled = (n["near_only"] * raw["near_only"] + n["both"] * raw["both"]) / (n["near_only"] + n["both"])
    assert np.allclose(d["caa_within"], pooled / np.linalg.norm(pooled), atol=1e-12)
    # The fixture's baseline chooses the correct option where there is one, so within near_only and within both
    # every prompt made the same choice: those contrasts cannot carry the planted choice vector c, and CAA does.
    # (c is rebuilt as fixtures.make_activations draws it. The fitted `choice` direction is not the yardstick: far
    # choices come with longer horizons, so it carries the planted horizon curve as well.)
    basis, _ = np.linalg.qr(np.random.default_rng(0).standard_normal((24, 4)))
    c = basis[:, 3]
    on_c = {key: float(d[key] @ c) for key in ("caa", "caa_none", "caa_near_only", "caa_both", "caa_within")}
    print(f"component along the planted choice vector: { {k: round(v, 3) for k, v in on_c.items()} }")
    assert all(abs(on_c[key]) < 0.05 for key in ("caa_near_only", "caa_both", "caa_within")) and on_c["caa"] > 0.2   # measured 0.281
    assert set(w["agreement"]) == {"both|near_only", "both|none", "near_only|none"}


def test_within_regime_settings_and_degenerate_regimes(tmp_path):
    acts = make_activations(tmp_path / "acts")
    half = run_fit(acts, tmp_path / "fits", "--name", "all3", "--within-pool", "none", "near_only", "both",
                   "--within-quantile", "0.25")
    w = json.loads((half / "fit.json").read_text())["diagnostics"]["caa_within"]
    print(f"quantile 0.25, pool of three: sizes { {g: (v['n_long'], v['n_short']) for g, v in w['by_regime'].items()} }")
    assert w["pool"] == ["none", "near_only", "both"] and w["quantile"] == 0.25
    assert all(v["n_long"] < 0.4 * v["n_regime"] for v in w["by_regime"].values())
    # A regime left with a single horizon has no long and short class: it is reported, not fitted, and not pooled.
    import numpy as np
    from steer.fitting import within_regime_classes
    t = np.array([1.0, 1.0, 1.0, 2.0, 3.0, 4.0])
    regime = np.array(["both"] * 3 + ["none"] * 3, dtype=object)
    classes, label = within_regime_classes(t, regime, 1 / 3)
    assert list(classes) == ["none"] and label.tolist() == ["", "", "", "short", "", "long"]
    with pytest.raises(ValueError):
        within_regime_classes(t, regime, 0.5)
