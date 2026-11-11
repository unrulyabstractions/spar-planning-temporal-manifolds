"""pipeline/s3_analysis/by_label/choice_structure.py on tiny chunk files with planted linear signals: the probes recover
a planted target, a target independent of the activations stays near chance, and a stopped run
with a handful of prompts is reported, not crashed on."""

import json
from pathlib import Path

import numpy as np
import pytest

import pipeline.s3_analysis.by_label.choice_structure as cs

K = 4
D = 16
LAYERS = (10, 14)
FRACTIONS = (0.25, 0.5, 1.0)
REGIONS = {"suffix": (0, 3), "think": (3, 6), "pre": (6, 8), "prelabel": (8, 9), "answer": (9, 15), "prompt": (15, 15 + 2 + 2 * K)}
LABELS = ["a", "b", "c", "d"]
PRELABEL, SUFFIX_NEWLINE, OPTION_LABEL = 8, 1, 17


def make_prompt(rng, i, base):
    """A record with 2 near and 2 far options in a random line order."""
    returns = {"near": rng.choice([2, 4, 6, 8, 10, 12], 2, replace=False), "far": rng.choice([50, 100, 200, 400], 2, replace=False)}
    options = [{"payout": 1000 + 10 * int(r), "return_pct": int(r), "delay_years": 0.5 if half == "near" else 10.0, "half": half,
                "rank_in_half": rank + 1} for half in ("near", "far") for rank, r in enumerate(sorted(returns[half], reverse=True))]
    rng.shuffle(options)
    for lab, o in zip(LABELS, options):
        o["label"] = lab
    regime = ("near", "both")[base % 2]
    truth_half = "near" if regime == "near" else "far"
    truth = next(o for o in options if o["half"] == truth_half and o["rank_in_half"] == 1)
    return {"horizon": float(10 ** rng.uniform(-1, 2)), "regime": regime, "truth": truth_half, "pair_id": f"p{base}",
            "duration_id": f"d{base % 7}", "rendered_unit": ("days", "weeks", "months")[i % 3], "options": options,
            "labels": LABELS, "near_labels": [o["label"] for o in options if o["half"] == "near"],
            "far_labels": [o["label"] for o in options if o["half"] == "far"], "truth_label": truth["label"]}


def write_planted_chunks(run_dir, n_chunks=4, per_chunk=24, seed=0):
    """Planted: the half chosen is written into pre-label along direction u (chosen near/far at random,
    so half chosen is not the regime), log horizon into the suffix newline along v, and each option's
    return into its label token along w. The truth position is independent of every activation."""
    rng = np.random.default_rng(seed)
    u, v, w = np.linalg.qr(rng.normal(size=(D, 3)))[0].T
    chunk_dir = Path(run_dir) / "chunks"
    chunk_dir.mkdir(parents=True)
    index = 0
    for c in range(n_chunks):
        rows, acts = [], {l: rng.normal(size=(per_chunk, 15 + 2 + 2 * K, D)).astype(np.float32) for l in LAYERS}
        order = np.zeros((per_chunk, K), int)
        for p in range(per_chunk):
            rec = make_prompt(rng, index, index // 3)
            near_pick = bool(rng.integers(2))
            pool = rec["near_labels"] if near_pick else rec["far_labels"]
            rec["label"] = str(rng.choice(pool))
            rec.update(answer=rec["label"], forced=bool(index % 3 == 0), think_len=5, think_text="x", prompt="p")
            rows.append(rec)
            order[p] = np.arange(K)
            for l in LAYERS:
                acts[l][p, PRELABEL] += 4 * (1 if near_pick else -1) * u
                acts[l][p, SUFFIX_NEWLINE] += 5 * np.log10(rec["horizon"]) * v
                for j, o in enumerate(rec["options"]):
                    acts[l][p, OPTION_LABEL + 2 * j] += 0.15 * o["return_pct"] * w
            index += 1
        np.savez(chunk_dir / f"chunk_{c:03d}.npz", layers=np.array(LAYERS), think_fractions=np.array(FRACTIONS),
                 option_order=order, forced_mask=np.array([r["forced"] for r in rows]),
                 **{f"region_{k}": np.array(x) for k, x in REGIONS.items()}, **{f"L{l}": a for l, a in acts.items()})
        (chunk_dir / f"chunk_{c:03d}.json").write_text(json.dumps({"rows": rows, "kept": [True] * per_chunk}))
    return rng


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("planted")
    write_planted_chunks(run_dir)
    return cs.analyze(run_dir, seconds=60), run_dir


def score(res, target, position, layer):
    return res["prompt_probes"][target][position][layer]["score"]


def test_positions_follow_the_kept_layout():
    index = cs.position_indices(REGIONS, FRACTIONS, K)
    assert index["suffix newline"] == 1 and index["think@1"] == 5 and index["think close"] == 6
    assert index["pre-label"] == 8 and index["phrase number"] == 15 and index["phrase unit"] == 16
    assert index["option 1 label"] == 17 and index["option 4 end"] == 24


def test_a_planted_half_choice_is_recovered_at_the_prelabel_and_not_at_an_unplanted_position(result):
    res, _ = result
    assert res["n"] == 96 and res["n_bases"] == 32 and not res["small_n"]
    assert score(res, "half_chosen", "pre-label", 10) > 0.9
    assert score(res, "half_chosen", "think@1", 10) < 0.7


def test_a_planted_log_horizon_is_recovered_by_regression(result):
    res, _ = result
    assert score(res, "log_horizon", "suffix newline", 14) > 0.8
    assert score(res, "log_horizon", "think close", 14) < 0.3


def test_targets_independent_of_the_activations_stay_near_chance(result):
    res, _ = result
    chance_cells = [res["prompt_probes"]["truth_pos"][p][l] for p in ("pre-label", "suffix newline") for l in LAYERS]
    assert all(0.1 < c["score"] < 0.42 and c["chance"] == 0.25 for c in chance_cells)


def test_planted_option_return_is_recovered_at_the_label_token_but_not_at_the_line_end(result):
    res, _ = result
    cells = res["option_probes"]["return_pct"]
    assert cells["label"][10]["score"] > 0.6 and cells["end"][10]["score"] < 0.2
    assert 0.35 < res["option_probes"]["is_chosen"]["label"][10]["score"] < 0.65


def test_outputs_are_written_and_the_report_names_n(result):
    res, run_dir = result
    out = run_dir / "choice_structure"
    assert {p.name for p in out.iterdir()} >= {"results.json", "results.txt", "probes_prompt_main.png",
                                               "probes_position.png", "probes_options.png"}
    saved = json.loads((out / "results.json").read_text())
    assert saved["n"] == 96 and saved["prompt_probes"]["half_chosen"]["pre-label"]["10"]["score"] > 0.9
    text = (out / "results.txt").read_text()
    assert "n = 96 kept prompts from 4 chunk(s), 32 bases" in text and "[half_chosen]" in text and "[option: return_pct]" in text


def test_forced_and_natural_splits_and_position_bias_are_reported(result):
    res, _ = result
    assert {"forced", "natural"} <= set(res["split"])
    assert res["split"]["natural"]["half_chosen"]["pre-label"][10]["score"] > 0.8
    assert abs(sum(res["position_bias"]["pick_rate"]) - 1) < 1e-9 and len(res["position_bias"]["truth_rate"]) == K


def test_horizon_pca_reports_unit_and_duration_shares(result):
    res, _ = result
    cell = res["horizon_pca"]["suffix newline"][10]
    assert cell["abs_r_log_horizon"][0] > 0.9 and len(cell["evr"]) == 3
    assert "unit_share" in cell and "duration_share" in cell


def test_a_stopped_run_with_a_few_prompts_is_reported_not_analysed(tmp_path):
    write_planted_chunks(tmp_path, n_chunks=1, per_chunk=4)
    res = cs.analyze(tmp_path)
    assert res["small_n"] and "prompt_probes" not in res
    text = (tmp_path / "choice_structure" / "results.txt").read_text()
    assert "n = 4 kept prompts" in text and "no probes were run" in text


def test_a_single_chunk_of_moderate_size_runs_without_crashing(tmp_path):
    write_planted_chunks(tmp_path, n_chunks=1, per_chunk=30)
    res = cs.analyze(tmp_path, seconds=60)
    assert res["n"] == 30 and not res["small_n"]
    assert (tmp_path / "choice_structure" / "results.txt").exists()
