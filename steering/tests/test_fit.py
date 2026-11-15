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
    assert sorted(p.name for p in out.iterdir()) == ["directions.npz", "fit.json", "manifold.npz", "per_horizon.jsonl"]
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
