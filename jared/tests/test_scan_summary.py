"""Part B scan summary (pipeline/s2_steering/analysis/scan_summary.py) on planted unit files: what counts as a flip, a
toward-target flip and an excess, which cells are reported, partial scans."""

import json

import pytest

import pipeline.s2_steering.analysis.scan_summary as summary

N = 10


def rows(labels, near_lo=-5.0, **extra):
    """Ten prompts whose near option is `a`; `labels` is the label each row got."""
    return [{"sample_uid": f"u{i}", "near_label": "a", "label": labels[i], "near_lo": near_lo, **extra}
            for i in range(N)]


def unit(name, labels, near_lo=-5.0, **extra):
    layer, position, direction = (name.split()[0][1:], name.split()[1], name.split()[2]) if name.startswith("L") \
        else (None, None, None)
    return {"name": name, "layer": layer and int(layer), "position": position, "direction": direction,
            "rows": rows(labels, near_lo, **extra)}


def write_scan(directory, units, layers=(18,), positions=("number", "unit")):
    (directory / "conditions").mkdir(parents=True)
    (directory / "inputs.json").write_text(json.dumps(
        {"layers": list(layers), "positions": list(positions), "directions": ["manifold", "random"]}))
    for u in units:
        (directory / "conditions" / f"{u['name'].replace(' ', '_')}.json").write_text(json.dumps(u))


def far(k_near):
    """Baseline-like labels: `b` (far) except the first `k_near` rows, which are `a` (near)."""
    return ["a"] * k_near + ["b"] * (N - k_near)


BASELINE = unit("baseline", far(0))


def test_flips_toward_and_away_count_against_the_baseline_label():
    base = rows(["a", "a", "b", "b", "b", "b", "b", "b", "b", "b"])
    cell = rows(["a", "b", "a", "a", "x", "b", "b", "b", "b", "b"], near_lo=-3.0)
    m = summary.flip_measures(cell, base)
    # row 1 near -> far (away), rows 2 and 3 far -> near (toward), row 4 far -> "x" (a flip, not toward)
    assert (m["flips"], m["toward"], m["away"]) == (4, 2, 2)
    assert m["near_lo_shift"] == pytest.approx(2.0)


def test_rows_with_their_own_reference_are_compared_with_it_not_the_baseline():
    base = rows(["b"] * N)
    answer = rows(["a", "a", "b"] + ["b"] * 7, near_lo=1.0,
                  ref_label="b", ref_near_lo=-1.0)
    answer[1]["ref_label"] = "a"
    m = summary.flip_measures(answer, base)
    assert (m["flips"], m["toward"]) == (1, 1)                  # only row 0; row 1 was already near
    assert m["near_lo_shift"] == pytest.approx(2.0)


def test_a_cell_is_reported_at_four_excess_toward_flips_next_to_its_twin_and_the_noise(tmp_path):
    noop1, noop2 = unit("noop 1", far(1)), unit("noop 2", far(0))             # mean 0.5 flips, 0.5 toward
    reported = unit("L18 number manifold", far(5))                          # 5 toward: excess 4.5
    twin = unit("L18 number random", far(1))
    short = unit("L18 unit manifold", far(4))                               # 4 toward: excess 3.5, not reported
    write_scan(tmp_path, [BASELINE, noop1, noop2, reported, twin, short])
    result = summary.summarize(tmp_path)
    assert (result["n_cells_done"], result["n_cells_total"], result["noops_done"]) == (3, 4, 2)
    by_name = {c["name"]: c for c in result["cells"]}
    assert by_name["L18 number manifold"]["excess_toward"] == pytest.approx(4.5)
    assert by_name["L18 number manifold"]["excess_total"] == pytest.approx(4.5)
    assert by_name["L18 unit manifold"]["excess_toward"] == pytest.approx(3.5)
    assert [item["cell"]["name"] for item in result["reported"]] == ["L18 number manifold"]
    item = result["reported"][0]
    assert item["twin"]["name"] == "L18 number random" and item["twin"]["flips"] == 1
    assert item["noise"]["mean"] == {"flips": 0.5, "toward": 0.5}
    text = summary.format_table(result)
    assert "cells done: 3 of 4" in text and "REPORTED" in text and "no significance claim" in text
    assert "twin L18 number random: 1 flips, 1 toward" in text


def test_flips_away_from_near_do_not_make_a_cell_reportable(tmp_path):
    base = unit("baseline", far(5))
    away = unit("L18 number manifold", far(0))                              # five flips, none toward near
    write_scan(tmp_path, [base, unit("noop 1", far(5)), unit("noop 2", far(5)), away])
    cell = next(c for c in summary.summarize(tmp_path)["cells"])
    assert (cell["flips"], cell["toward"], cell["excess_total"], cell["excess_toward"]) == (5, 0, 5.0, 0.0)
    assert not cell["reported"]


def test_partial_scans_report_what_exists(tmp_path):
    write_scan(tmp_path, [])
    assert summary.summarize(tmp_path)["baseline_done"] is False
    assert "MISSING" in summary.format_table(summary.summarize(tmp_path))

    write_scan(tmp_path / "b", [BASELINE, unit("L18 number manifold", far(5))])
    result = summary.summarize(tmp_path / "b")
    assert result["noise"] is None and result["reported"] == [] and result["n_cells_done"] == 1
    assert "no excess can be computed" in summary.format_table(result)


def test_main_writes_json_and_text(tmp_path, capsys):
    write_scan(tmp_path, [BASELINE, unit("noop 1", far(0)), unit("noop 2", far(0)),
                          unit("L18 number manifold", far(6))])
    summary.main(tmp_path)
    saved = json.loads((tmp_path / "scan_summary.json").read_text())
    assert saved["reported"][0]["cell"]["name"] == "L18 number manifold"
    text = (tmp_path / "scan_summary.txt").read_text()
    assert capsys.readouterr().out == text + "\n" and "REPORTED" in text
