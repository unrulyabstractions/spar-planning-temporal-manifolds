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
