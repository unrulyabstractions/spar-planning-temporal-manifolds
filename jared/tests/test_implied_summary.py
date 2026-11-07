"""E18 summary on synthetic rows built from the real bank: the stated-horizon reader, behaviour measures,
criteria C0-C4, the probe readout with a planted signal, the audit sample, and an empty run."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
import implied_summary as S  # noqa: E402
from spar_horizon.full_bank import natural_horizon  # noqa: E402
from spar_horizon.implied_bank import build_implied_prompts  # noqa: E402

D = 8
LAYERS = (16, 20, 24)


@pytest.fixture(scope="module")
def bank():
    return build_implied_prompts()


def far_label(rec):
    return "b" if rec["near_label"] == "a" else "a"


def rational(rec):
    return rec["truth"] == "near"


def states_horizon(rec):
    return f"The time horizon is about {natural_horizon(10 ** rec['log_h'])}." if rec["log_h"] is not None else "No horizon."


def write_run(path, records, chooses_near=rational, think=states_horizon, state=None, forced=lambda r: False):
    """Two units of fake rows (and states when `state(rec)` gives a vector per column)."""
    unit_dir = path / "units"
    unit_dir.mkdir(parents=True)
    half = len(records) // 2
    for n, part in enumerate((records[:half], records[half:])):
        rows = []
        for r in part:
            row = {k: v for k, v in r.items() if k != "prompt"}
            row.update(answer="a) x", label=r["near_label"] if chooses_near(r) else far_label(r), forced=forced(r),
                       think_len=9, think_text=think(r), label_logits=[0.0, 0.0])
            rows.append(row)
        (unit_dir / f"M1_{n:03d}.json").write_text(json.dumps({"unit": f"M1_{n:03d}", "rows": rows}))
        arrays = {"layers": np.array(LAYERS), "cols": np.array(["suffix_0", "suffix_1", "think@1", "prelabel"]),
                  "sample_uids": np.array([r["sample_uid"] for r in part])}
        for layer in LAYERS:
            arrays[f"L{layer}"] = np.stack([state(r) if state else np.zeros((4, D)) for r in part]).astype(np.float32)
        np.savez(unit_dir / f"M1_{n:03d}.npz", **arrays)
    return path


def keep_modules(bank, *modules):
    return [r for r in bank if r["module"] in modules]


@pytest.mark.parametrize("text,years", [
    ("the lease expires in 18 months, so", 1.5),
    ("That is about 7 hours of waiting.", 7 / 8766),
    ("The time horizon is 2.5 years here.", 2.5),
    ("a 13-year time horizon", 13),
    ("so the horizon given is roughly three weeks", 3 * 7 / 365.25),
    ("A generation is approximately 25 years.", 25),
])
def test_the_reader_finds_a_duration_stated_as_the_horizon(text, years):
    found = S.first_stated_implied(text, 100.0, 200.0)
    assert found is not None and found[0] == pytest.approx(years, rel=1e-3)


def test_the_reader_skips_the_prompts_own_delays_and_durations_with_no_connective():
    assert S.first_stated_implied("Option a pays in 6 months and b in 3 years.", 0.5, 3.0) is None
    assert S.first_stated_implied("5,000 dollars, 18 months later we see", 100.0, 200.0) is None
    found = S.first_stated_implied("b pays in 3 years, but the lease ends in 18 months.", 0.5, 3.0)
    assert found[0] == pytest.approx(1.5)


def test_a_rational_model_that_states_the_truth_passes_c0_to_c2(bank, tmp_path):
    run = write_run(tmp_path / "run", keep_modules(bank, "M1", "M2"))
    summary, _ = S.summarize(run)
    c = summary["criteria"]
    assert c["C0"] == "pass" and c["C1"] == {"A": "pass", "B": "pass", "C": "pass"}
    assert c["C2"]["overall"] == "pass" and c["C2"]["parts"]["B"] == "pass"
    assert c["C3"] == "pending" and c["C4"] == "pending"
    assert summary["behaviour"]["A"]["implied near+both"]["correct"] == summary["behaviour"]["A"]["implied near+both"]["n"] == 32
    assert summary["conversion"]["C"]["median_error_decades"] < 0.05 and summary["conversion"]["B"]["inside_share"] == 1.0
    assert summary["behaviour"]["A"]["mcnemar"] == {"pairs": 32, "only_implied_correct": 0,
                                                    "only_explicit_correct": 0, "p": None}


def test_a_model_that_ignores_the_anchor_fails_c1_as_ignored(bank, tmp_path):
    ignores = lambda r: True if r["rendering"] in ("implied", "null") else rational(r)  # noqa: E731
    summary, _ = S.summarize(write_run(tmp_path / "run", keep_modules(bank, "M1", "M2"), ignores))
    assert summary["criteria"]["C0"] == "pass"
    assert summary["criteria"]["C1"] == {"A": "fail (ignored)", "B": "fail (ignored)", "C": "fail (ignored)"}
    assert summary["behaviour"]["A"]["mcnemar"]["only_explicit_correct"] == 16


def test_c1_fails_without_being_ignored_when_implied_is_wrong_but_not_like_the_null(bank, tmp_path):
    inverted = lambda r: (not rational(r)) if r["rendering"] == "implied" else (r["rendering"] == "null" or rational(r))  # noqa: E731
    summary, _ = S.summarize(write_run(tmp_path / "run", keep_modules(bank, "M1"), inverted))
    assert summary["criteria"]["C1"]["A"] == "fail"


def test_c0_fails_when_the_twin_is_not_accurate(bank, tmp_path):
    wrong_twin = lambda r: (not rational(r)) if r["rendering"] == "explicit" and r["regime"] == "both" else rational(r)  # noqa: E731
    summary, _ = S.summarize(write_run(tmp_path / "run", keep_modules(bank, "M1"), wrong_twin))
    assert summary["criteria"]["C0"] == "fail" and summary["criteria"]["C0 twin accuracy"]["both"] == 0


def test_natural_closes_are_reported_beside_all_rows(bank, tmp_path):
    forced = lambda r: r["rendering"] == "implied" and r["regime"] == "both"  # noqa: E731
    summary, _ = S.summarize(write_run(tmp_path / "run", keep_modules(bank, "M1"), forced=forced))
    arms = summary["behaviour"]["A"]
    assert arms["implied both"]["forced"] == 1.0 and arms["implied both natural"]["n"] == 0
    assert arms["implied near natural"]["n"] == arms["implied near"]["n"]


def misstated(rec):
    """A trace stating a duration in the other regime of near/both."""
    near, far = rec["near_delay_years"], rec["far_delay_years"]
    years = far * 10 if rec["regime"] == "near" else (near * far) ** 0.5
    return f"The time horizon is about {natural_horizon(years)}."


def test_mediation_counts_regime_flipping_misconversions_and_the_share_following_them(bank, tmp_path):
    records = keep_modules(bank, "M1")
    follows = lambda r: (not rational(r)) if r["rendering"] == "implied" else rational(r)  # noqa: E731
    think = lambda r: misstated(r) if r["rendering"] == "implied" else states_horizon(r)  # noqa: E731
    summary, _ = S.summarize(write_run(tmp_path / "followed", records, follows, think))
    assert summary["mediation"]["flips"] == 64 and summary["mediation"]["follows_share"] == 1.0
    assert summary["criteria"]["C4"] == "pass"
    summary, _ = S.summarize(write_run(tmp_path / "ignored", records, rational, think))
    assert summary["mediation"]["follows_share"] == 0.0 and summary["criteria"]["C4"] == "fail"
    assert summary["conversion"]["A"]["flips_regime"] == 32


def test_mediation_with_fewer_than_ten_flips_is_underpowered(bank, tmp_path):
    records = keep_modules(bank, "M1")
    some = [r["sample_uid"] for r in records if r["rendering"] == "implied"][:5]
    think = lambda r: misstated(r) if r["sample_uid"] in some else states_horizon(r)  # noqa: E731
    summary, _ = S.summarize(write_run(tmp_path / "run", records, rational, think))
    assert summary["mediation"]["flips"] == 5 and summary["criteria"]["C4"] == "underpowered"


def test_a_trace_that_states_nothing_is_not_converted(bank, tmp_path):
    summary, _ = S.summarize(write_run(tmp_path / "run", keep_modules(bank, "M1"), think=lambda r: "Hmm, let me think."))
    assert summary["conversion"]["A"]["converted"] == 0 and summary["criteria"]["C2"]["overall"] == "fail"


def make_probe_run(path, rng, direction, n_horizons=20, per=6):
    """An explicit E0-style run whose states at every position carry log10 horizon along `direction`."""
    path.mkdir(parents=True)
    log_h = np.repeat(np.linspace(-3, 2, n_horizons), per)
    n, positions = len(log_h), 12
    arrays = {"horizons": 10 ** log_h, "forced_mask": np.zeros(n, bool), "think_fractions": np.array([0.25, 1.0]),
              "region_suffix": np.array([0, 2]), "region_think": np.array([2, 4]), "region_pre": np.array([4, 6]),
              "region_prelabel": np.array([6, 7]), "region_answer": np.array([7, 12])}
    for layer in LAYERS:
        X = rng.normal(size=(n, positions, D)) * 0.05 + log_h[:, None, None] * direction
        arrays[f"L{layer}"] = X.astype(np.float32)
    np.savez(path / "activations.npz", **arrays)
    return path


@pytest.mark.parametrize("implied_has_signal,verdict", [(True, "present"), (False, "absent")])
def test_readout_compares_implied_and_explicit_states_with_the_explicit_trained_probe(bank, tmp_path, implied_has_signal,
                                                                                    verdict):
    rng = np.random.default_rng(0)
    direction = rng.normal(size=D)
    probe = make_probe_run(tmp_path / "e0", rng, direction)
    noise_seed = np.random.default_rng(1)

    def state(rec):
        if rec["rendering"] == "implied" and not implied_has_signal:
            return noise_seed.normal(size=(4, D)) * 2
        return np.full((4, D), rec["log_h"] if rec["log_h"] is not None else 0.0) * direction + noise_seed.normal(size=(4, D)) * 0.05

    records = [r for r in keep_modules(bank, "M1") if r["rendering"] in ("implied", "explicit")]
    summary, _ = S.summarize(write_run(tmp_path / "run", records, state=state), [probe])
    cell = summary["readout"]["suffix_1 L16 A"]
    assert cell["verdict"] == verdict, cell
    assert cell["explicit"]["mae"] < 0.3 and cell["explicit"]["rho"] > 0.9 and cell["explicit"]["n"] == 32
    assert set(summary["readout"]) >= {f"{c} L{l} {f}" for c, ls in S.CELLS for l in ls for f in "AC"}
    assert summary["criteria"]["C3"]["suffix_1 L16 A"] == verdict


def test_readout_is_pending_without_probe_runs(bank, tmp_path):
    summary, _ = S.summarize(write_run(tmp_path / "run", keep_modules(bank, "M1")))
    assert summary["readout"] == {} and "no --probe-runs given" in S.format_report(summary)


def test_audit_sample_holds_forty_implied_traces_without_automatic_coding(bank, tmp_path):
    summary, items = S.summarize(write_run(tmp_path / "run", keep_modules(bank, "M1", "M2")))
    assert len(items) == 40 and {i["family"] for i in items} == {"A", "B", "C"}
    assert all(set(i) == {"id", "family", "context", "constraint_phrase", "forced", "think_text"} for i in items)
    assert [i["id"] for i in items] == list(range(40)) and len(summary["audit_key"]) == 40
    assert all(k["auto_stated"] is not None for k in summary["audit_key"])
    again = S.summarize(tmp_path / "run")[1]
    assert again == items


def test_an_empty_run_reads_pending_and_main_writes_the_three_files(tmp_path):
    S.main(tmp_path)
    summary = json.loads((tmp_path / "implied_summary.json").read_text())
    assert summary["criteria"]["C0"] == "pending" and summary["criteria"]["C4"] == "pending"
    assert summary["criteria"]["C1"] == {"A": "pending", "B": "pending", "C": "pending"}
    assert json.loads((tmp_path / "implied_audit_sample.json").read_text()) == {"items": []}
    assert "E18 implied horizons: 0 rows" in (tmp_path / "implied_summary.txt").read_text()
