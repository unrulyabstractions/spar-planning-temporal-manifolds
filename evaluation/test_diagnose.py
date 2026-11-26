"""Tests for diagnose.py on synthetic runs (CPU, no model). Run (from evaluation/): python -m pytest -q test_diagnose.py"""

import json
from html.parser import HTMLParser

import numpy as np
import pytest

import diagnose as D

N, DIM = 240, 48


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def make_artifacts(root, letter_signal, seed=0):
    """Fit rows whose CAA class is the first-listed answer; X is noise plus letter_signal x the letter direction."""
    rng = np.random.default_rng(seed)
    first = rng.random(N) < 0.5
    rows = [{"uid": f"u{i}", "regime": ("none", "near_only", "both")[i % 3], "caa_class": "long" if first[i] else "short",
             "caa_within_class": rng.choice(["long", "short"]), "reward_class": rng.choice(["high", "low"]),
             "chosen_option": rng.choice(["far", "near"]),     # independent of the class: equal far rates
             "choice_label": "a)" if first[i] else "b)", "label_style": rng.choice(["a)/b)", "1)/2)"]),
             "near_first": bool(rng.random() < 0.5), "fold": i % 5} for i in range(N)]
    letter_dir = rng.normal(size=DIM)
    X = rng.normal(size=(N, DIM)) + letter_signal * np.where(first, 1.0, -1.0)[:, None] * letter_dir
    acts, bundle = root / "activations", root / "methods" / "caa__test"
    bundle.mkdir(parents=True)
    acts.mkdir()
    write_jsonl(acts / "index.jsonl", [{"row": i, "uid": r["uid"]} for i, r in enumerate(rows)])
    np.save(acts / "L1_label.npy", X.astype(np.float32))
    (bundle / "method.json").write_text(json.dumps({"method": "caa", "kind": "push", "site": "label", "layer": 1}))
    write_jsonl(bundle / "fit_uids.jsonl", rows)
    long, short = X[first], X[~first]
    np.savez(bundle / "arrays.npz", direction=long.mean(0) - short.mean(0), typical_norm=np.float64(10.0))
    return root


def run_diagnose(tmp_path, *args):
    out = tmp_path / "out"
    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    D.main([str(run), "--out", str(out), *args])
    return out, json.loads((out / "diagnostics.json").read_text())


def flagged(report, section, name_part):
    hits = [c for c in report["sections"][section]["checks"] if name_part in c["name"]]
    assert hits, name_part
    return all(c["flag"] for c in hits)


def test_letter_direction_flagged_on_letter_and_equal_far_rates_on_gap(tmp_path):
    art = make_artifacts(tmp_path / "art", letter_signal=1.0)
    _, report = run_diagnose(tmp_path, "--artifacts", str(art))
    assert flagged(report, "C", "caa: cos with letter")
    assert flagged(report, "C", "caa [all]: far-rate gap")
    assert not flagged(report, "C", "caa: saved bundle vs recomputed")      # the saved direction is the recomputed one


def test_pure_noise_direction_flagged_on_reliability(tmp_path):
    art = make_artifacts(tmp_path / "art", letter_signal=0.0)
    _, report = run_diagnose(tmp_path, "--artifacts", str(art))
    assert flagged(report, "C", "caa: split-half reliability")


def test_peaked_and_reversing_dose_curve_with_x_past_peak_is_flagged(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    odd = [0.1, 0.4, 0.7, 0.5, -1.0]
    sweep = [{"x": x, "mean_odd_logit": e, "even_label_mass": 0.99, "healthy": True} for x, e in zip([0.05, 0.1, 0.2, 0.4, 0.8], odd)]
    (run / "calibration.json").write_text(json.dumps({"sweep": sweep, "x_push": 0.8, "rule": "test"}))
    _, report = run_diagnose(tmp_path)
    assert flagged(report, "D", "past the dose peak")
    assert flagged(report, "D", "changes sign")
    assert report["sections"]["D"]["data"]["peak_x"] == 0.2


def test_rising_dose_curve_is_not_flagged(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    sweep = [{"x": x, "mean_odd_logit": e, "even_label_mass": 0.99, "healthy": True} for x, e in [(0.1, 0.2), (0.2, 0.5), (0.4, 1.0)]]
    (run / "calibration.json").write_text(json.dumps({"sweep": sweep, "x_push": 0.4, "rule": "test"}))
    _, report = run_diagnose(tmp_path)
    assert not any(c["flag"] for c in report["sections"]["D"]["checks"])


def test_missing_inputs_give_skipped_sections(tmp_path):
    out, report = run_diagnose(tmp_path)
    assert all(report["sections"][s]["skipped"].startswith("skipped:") for s in "ABCDEF")
    assert "skipped:" in (out / "diagnostics.html").read_text()


def test_html_is_well_formed_and_has_no_external_urls(tmp_path):
    art = make_artifacts(tmp_path / "art", letter_signal=1.0)
    run = tmp_path / "run"
    run.mkdir()
    sweep = [{"x": x, "mean_odd_logit": e, "even_label_mass": 0.99, "healthy": True} for x, e in [(0.1, 0.2), (0.2, 0.5), (0.4, -1.0)]]
    (run / "calibration.json").write_text(json.dumps({"sweep": sweep, "x_push": 0.4, "rule": "test"}))
    out, _ = run_diagnose(tmp_path, "--artifacts", str(art))
    page = (out / "diagnostics.html").read_text()
    assert "http" not in page and "<script" not in page and "<svg" in page and "prefers-color-scheme: dark" in page

    stack, void = [], {"meta", "br", "hr", "img", "input", "link"}

    class Balanced(HTMLParser):
        def handle_starttag(self, tag, attrs):
            if tag not in void:
                stack.append(tag)

        def handle_startendtag(self, tag, attrs):
            pass

        def handle_endtag(self, tag):
            assert stack and stack.pop() == tag, tag

    Balanced().feed(page)
    assert stack == []
    assert 2 <= len(page.split("<title>")[1].split("</title>")[0].split()) <= 4


def test_fisher_p_matches_a_known_value():
    assert D.fisher_p(8, 2, 1, 5) == pytest.approx(0.0350, abs=1e-3)


def test_accuracy_breakdown_flags_the_odd_phrasing():
    rows = []
    for phrasing, correct in [(0, 24), (1, 6), (2, 25), (3, 24)]:
        for i in range(30):
            rows.append({"regime": "both", "phrasing_id": phrasing, "labels": ["a)", "b)"], "near_first": i % 2 == 0,
                         "correct_option": "far", "chosen_option": "far" if i < correct else "near"})
    _, checks, _ = D.accuracy_breakdown(rows)
    odd = {c["name"]: c["flag"] for c in checks if "phrasing" in c["name"]}
    looks = [n for n, f in odd.items() if f]
    assert any("phrasing = 1," in n for n in looks)
    assert not any("phrasing = 0," in n or "phrasing = 3," in n for n in looks)
    assert not any(c["flag"] for c in checks if "label style" in c["name"] or "option order" in c["name"])
