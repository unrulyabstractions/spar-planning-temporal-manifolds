"""Tests for the transfer stage (no GPU, no model): conversation rendering, the baseline with a fake model, the replay
stage on the transfer split with fake edits and readout, and the per-type report on hand-made arm results.
Run (from evaluation/): python -m pytest -q test_transfer_eval.py"""

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "prompt_bank"))   # pbank, when not installed

import analyze
import replay_eval as rv
import transfer_eval as te

MESSAGES = [{"role": "system", "content": "You must select ... time horizon: 2 months."},
            {"role": "user", "content": "a) 1,000 dollars in 1 week.\nb) 4,000 dollars in 1 year.\nFORMAT: I choose:"}]


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", local_files_only=True)
    except Exception:
        pytest.skip("Qwen3-1.7B tokenizer not cached")


@pytest.mark.parametrize("thinking", [True, False])
def test_chat_text_matches_pbank_for_a_user_text(tok, thinking):
    from pbank.local_model import LocalChatModel
    text = "SITUATION: Plan for the future of the household."
    assert te.chat_text(tok, text, thinking) == LocalChatModel.chat_text(SimpleNamespace(tok=tok), text, thinking)


def test_chat_text_renders_the_whole_conversation(tok):
    s = te.chat_text(tok, MESSAGES, False)
    assert s.index(MESSAGES[0]["content"]) < s.index(MESSAGES[1]["content"])
    one_turn = te.chat_text(tok, "SENTINEL", False)
    assert s.endswith(one_turn[one_turn.index("SENTINEL") + len("SENTINEL"):])   # same turn end and generation prompt


def row(uid, regime="both", near_first=True, messages=None, variant="base", pair=None, correct="far", r=2.0):
    """A bank row in the exporter's layout (pbank PromptRecord fields, eval-type record in notes)."""
    notes = {"type_id": "toy", "variant": variant, "twin_id": pair or uid}
    if messages:
        notes["messages"] = messages
    labels = ["a)", "b)"]
    return {"uid": uid, "tier": "evaltype", "split": "transfer", "text": f"prompt {uid}", "horizon": "2 years",
            "horizon_grid_value": "2 years", "horizon_unit": "years", "horizon_years": 2.0, "rewritten": False,
            "phrasing_id": None, "near": {"reward": 1000, "delay": "1 week", "delay_years": 0.02},
            "far": {"reward": 4000, "delay": "1 year", "delay_years": 1.0}, "reward_multiplier": 4.0,
            "near_first": near_first, "labels": labels, "regime": regime, "correct_option": correct,
            "correct_label": None if correct is None else labels[(correct == "far") == near_first],
            "horizon_equals_a_delay": False, "h_over_far_delay": r, "pair_id": pair or uid, "notes": notes}


def test_model_input():
    assert te.model_input(row("u1")) == "prompt u1"
    assert te.model_input(row("u2", messages=MESSAGES)) == MESSAGES


class FakeChat:
    def __init__(self):
        self.seen = []

    def generate(self, prompts, thinking, max_new_tokens):
        self.seen += prompts
        return [SimpleNamespace(thinking="t", answer="I choose: b). My reasoning: x", n_generated=5,
                                hit_token_cap=False, seconds=0.1, generated_ids=[1, 2, 3]) for _ in prompts]


SETTINGS = {"model_id": "toy/model", "thinking": True, "max_new_tokens": 64, "batch_size": 2}


def write_bank(d: Path, rows):
    d.mkdir(parents=True, exist_ok=True)
    rv.write_jsonl_atomic(d / "prompts.jsonl", rows)


def test_baseline_runs_scored_prompts_with_conversations(tmp_path):
    bank = tmp_path / "bank"
    write_bank(bank, [row("u1"), row("u2", messages=MESSAGES), row("u3", regime="none", correct=None),
                      row("u4", regime="near_only", correct="near")])
    chat = FakeChat()
    te.stage_baseline(bank, tmp_path / "run", chat, SETTINGS)
    assert chat.seen == ["prompt u1", MESSAGES, "prompt u4"]               # the none-regime prompt is not run
    resp = {r["uid"]: r for r in rv.read_jsonl(tmp_path / "run" / "responses.jsonl")}
    assert resp["u1"]["choice_label"] == "b)" and resp["u1"]["chosen_option"] == "far" and resp["u1"]["correct"]
    assert resp["u4"]["chosen_option"] == "far" and resp["u4"]["correct"] is False
    assert resp["u2"]["generated_ids"] == [1, 2, 3] and resp["u2"]["parse_status"] == "ok"
    assert rv.run_settings(tmp_path / "run")["model_id"] == "toy/model"     # what the items stage reads
    te.stage_baseline(bank, tmp_path / "run", chat, SETTINGS)                # done: nothing regenerated
    assert len(chat.seen) == 3
    with pytest.raises(SystemExit):
        te.stage_baseline(bank, tmp_path / "run", chat, {**SETTINGS, "thinking": False})


def test_baseline_hash_covers_the_conversation():
    other = [dict(MESSAGES[0], content="... time horizon: 20 years."), MESSAGES[1]]
    a, b = row("u2", messages=MESSAGES), row("u2", messages=other)
    assert a["text"] == b["text"] and te.inputs_sha256([a]) != te.inputs_sha256([b])


def test_replayer_renders_the_conversation():
    rep = object.__new__(te.TransferReplayer)
    seen = []
    rep.M = SimpleNamespace(prompt_ids=lambda p, thinking: seen.append(p) or [7])
    rep.bank = {"u2": row("u2", messages=MESSAGES)}
    rep.resp = {"u2": {"generated_ids": [1, 2, 3, 4]}}
    rep.thinking, rep._cache = True, {}
    assert rep.seqs({"uid": "u2", "label_index": 2}) == ([7], [1, 2])
    assert seen == [MESSAGES]


# ------------------------------------------------------------------ replay on the transfer split, with fakes

class FakeFactory:
    """Edits are named by arm; the readout adds x × the arm's effect to the unsteered y."""

    def __init__(self, args):
        self.caa = SimpleNamespace(edit="caa")
        self.positions = "all"

    def make(self, arm):
        return arm["arm"], "all"


class FakeReplayer:
    EFFECT = {"caa": 2.0}

    def __init__(self):
        self.calls = []

    def run(self, items, edit, x, positions="all"):
        self.calls.append((edit, x, [it["uid"] for it in items]))
        y = np.array([it["y0"] for it in items], np.float32) + x * self.EFFECT.get(edit, 0.1)
        return y, np.full(len(items), 0.9, np.float32)

    def plain(self, it):
        import torch
        lg = torch.zeros(10)
        lg[it["far_id"]] = it["y0"]
        return lg


def test_replay_stage_on_the_transfer_split(tmp_path, monkeypatch):
    monkeypatch.setattr(rv, "EditFactory", FakeFactory)
    methods = tmp_path / "methods"
    for m in ("caa", "curve-displace"):
        (methods / m).mkdir(parents=True)
        (methods / m / "method.json").write_text(json.dumps({"method": m}))
    out = tmp_path / "out"
    out.mkdir()
    items = [{"uid": f"t{i}", "split": "transfer", "far_id": 1, "near_id": 2, "y0": float(i), "chosen": "far"}
             for i in range(4)]
    items.append({"uid": "v0", "split": "verification", "far_id": 1, "near_id": 2, "y0": 0.0, "chosen": "near"})
    rv.write_jsonl_atomic(out / "items.jsonl", items)
    ns = SimpleNamespace(bank="t0", activations="a", methods=str(methods), draws="d", k=2, n0_check=2, out=str(out))
    rep = FakeReplayer()
    rv.stage_replay(ns, None, SETTINGS, rep, {"x_push": 0.3, "x_shift": 1.0}, split="transfer")
    assert all(set(uids) <= {"t0", "t1", "t2", "t3"} for _, _, uids in rep.calls)   # never the T0 grid's items
    assert ("caa", 0.3, ["t0", "t1", "t2", "t3"]) in rep.calls and ("caa", -0.3, ["t0", "t1", "t2", "t3"]) in rep.calls
    arms = rv.read_jsonl(out / "arms.jsonl")
    assert {"caa", "isotropic[1]", "in_span[0]", "shuffled:caa[1]"} <= {a["arm"] for a in arms}
    caa = np.load(out / "arms" / (hashlib.sha1(b"caa").hexdigest()[:12] + ".npz"))
    assert np.allclose(caa["y_plus"] - caa["y_minus"], 2 * 0.3 * 2.0)
    assert json.loads((out / "n0_check.json").read_text())["max_abs_diff_vs_hook_free"] == pytest.approx(0, abs=1e-6)


# ------------------------------------------------------------------ report on hand-made arm results

K = 19


def write_out(out: Path, items, effects: dict, split: str):
    """A replay_eval output directory: items, the unsteered pass and arm arrays with y(±x) = y0 ± effect."""
    (out / "arms").mkdir(parents=True)
    rv.write_jsonl_atomic(out / "items.jsonl", [dict(it, split=split) for it in items])
    rv.write_jsonl_atomic(out / "skipped.jsonl", [])
    logr = np.log10([it["h_over_far_delay"] for it in items])
    y0 = 6 * logr
    np.savez(out / "arms" / "_none.npz", y=y0, mass=np.full(len(items), 0.8))
    arms = [{"arm": "caa", "family": "method", "kind": "push", "x": 0.3}]
    arms += [{"arm": f"isotropic[{i}]", "family": "isotropic", "kind": "push", "x": 0.3, "row": i} for i in range(K)]
    with open(out / "arms.jsonl", "w") as f:
        for a in arms:
            f.write(json.dumps(a) + "\n")
    for a in arms:
        e = effects.get(a["arm"], 0.01 * (int(a["arm"][10:-1]) + 1) if a["family"] == "isotropic" else 0.0)
        e = e(items) if callable(e) else e
        np.savez(out / "arms" / (hashlib.sha1(a["arm"].encode()).hexdigest()[:12] + ".npz"),
                 y_plus=y0 + e, y_minus=y0 - e, mass_plus=np.full(len(items), 0.7), mass_minus=np.full(len(items), 0.75))


def grid_items(n_pairs, prefix, no_far_truth=False):
    r = np.logspace(np.log10(1 / 2.8), np.log10(2.8), n_pairs)
    out = []
    for i in range(n_pairs):
        correct = "far" if r[i] > 1 else "near"
        if no_far_truth and correct == "far":
            correct = None
        for nf in (True, False):
            out.append({"uid": f"{prefix}{i}-{nf}", "pair_id": f"{prefix}{i}", "near_first": nf,
                        "h_over_far_delay": float(r[i]), "correct_option": correct})
    return out


def type_bank(items, twins_every=4):
    """Bank rows for items: every twins_every-th pair's near-first member is an explicit twin with its own pair_id."""
    rows = []
    for n, it in enumerate(items):
        is_twin = it["near_first"] and (n // 2) % twins_every == 0
        if is_twin:
            it["pair_id"] = it["uid"]
        rows.append(row(it["uid"], variant="twin" if is_twin else ("base" if it["near_first"] else "swap"),
                        pair=it["pair_id"], correct=it["correct_option"], r=it["h_over_far_delay"]))
    return rows


def make_runs(tmp_path):
    """A T0 run where CAA moves y by 1, and two types where it moves the prompts by 0.5 and the twins by 1."""
    t0 = tmp_path / "t0"
    write_out(t0, grid_items(30, "v"), {"caa": 1.0}, "verification")
    (t0 / "calibration.json").write_text(json.dumps({"x_push": 0.3, "x_shift": 1.0}))
    banks, out = tmp_path / "banks", tmp_path / "transfer"
    for t, no_far in (("toy", False), ("novalues", True)):
        items = grid_items(20, t, no_far_truth=no_far)
        write_bank(banks / t, type_bank(items))
        twin = {r["uid"] for r in type_bank(items) if r["notes"]["variant"] == "twin"}
        write_out(out / t, items, {"caa": lambda its: np.array([1.0 if it["uid"] in twin else 0.5 for it in its])},
                  "transfer")
    return t0, banks, out


def test_report_per_type_and_group(tmp_path):
    t0, banks, out = make_runs(tmp_path)
    args = SimpleNamespace(out=str(out), t0_out=str(t0), banks=str(banks))
    rep = te.stage_report(args, ["toy", "novalues", "absent"], {"absent": "no bank"})

    assert rep["frozen"]["x_push"] == 0.3
    g0 = rep["t0"]
    assert g0["arms"]["caa"]["meaning"] == pytest.approx(1.0) and g0["switch_applies"]
    assert g0["comparisons"]["caa"]["isotropic"]["meaning"]["p"] == pytest.approx(1 / (K + 1))
    assert g0["unsteered"]["accuracy"] == 1.0 and g0["unsteered"]["label_mass"] == pytest.approx(0.8)

    prompts, twins = rep["types"]["toy"]["prompts"], rep["types"]["toy"]["twins"]
    assert prompts["n_items"] + twins["n_items"] == 40 and twins["n_items"] == 5
    assert prompts["arms"]["caa"]["meaning"] == pytest.approx(0.5)            # on the base/swap pairs
    assert "meaning" not in twins["arms"]["caa"]                               # twins have no order partner
    assert twins["arms"]["caa"]["odd_mean"] == pytest.approx(1.0)
    assert twins["comparisons"]["caa"]["isotropic"]["odd_mean"]["p"] == pytest.approx(1 / (K + 1))
    assert prompts["switch_applies"] and "delta_log10_H" in prompts["comparisons"]["caa"]["isotropic"]
    assert prompts["arms"]["caa"]["label_mass_plus"] == pytest.approx(0.7)

    nv = rep["types"]["novalues"]["prompts"]
    assert not nv["switch_applies"] and nv["n_correct_far"] == 0
    assert "delta_log10_H" not in nv["arms"]["caa"] and nv["unsteered"]["switch_loc"] is None
    assert "delta_log10_H" not in nv["comparisons"]["caa"]["isotropic"]

    assert rep["skipped_types"] == {"absent": "no bank"}
    md = (out / "transfer_report.md").read_text()
    assert "| T0 verification grid | grid |" in md and "| toy | twins |" in md and "absent (no bank)" in md


def test_main_report_only(tmp_path):
    """The command line, report stage only (no model): types found under --banks, settings written."""
    t0, banks, out = make_runs(tmp_path)
    (tmp_path / "t0run").mkdir()
    (tmp_path / "t0run" / "run.json").write_text(json.dumps({"settings": {**SETTINGS, "model": "toy/model"}}))
    common = ["--banks", str(banks), "--t0-bank", "b", "--t0-run", str(tmp_path / "t0run"), "--activations", "a",
              "--methods", "m", "--draws", "d", "--out", str(out)]
    te.main(["report", "--t0-out", str(t0), *common])
    rep = json.loads((out / "transfer_report.json").read_text())
    assert set(rep["types"]) == {"toy", "novalues"} and not rep["skipped_types"]
    assert json.loads((out / "settings.json").read_text())["calibration"]["x_push"] == 0.3
    with pytest.raises(SystemExit, match="calibration.json missing"):
        te.main(["report", "--t0-out", str(tmp_path / "nowhere"), *common])


def test_analyze_main_unchanged_on_t0(tmp_path):
    """analyze.py's report card, now built from arm_results and compare_to_nulls, on a hand-made T0 run."""
    t0 = tmp_path / "t0"
    write_out(t0, grid_items(30, "v"), {"caa": 1.0}, "verification")
    (t0 / "calibration.json").write_text(json.dumps({"x_push": 0.3, "x_shift": 1.0, "fold": 0, "target_logit": 1.0,
                                                     "reached_target": True}))
    analyze.main(["--out", str(t0)])
    rep = json.loads((t0 / "report.json").read_text())
    assert rep["comparisons"]["caa"]["isotropic"]["meaning"]["p"] == pytest.approx(1 / (K + 1))
    assert set(rep["comparisons"]["caa"]["isotropic"]) == {"meaning", "delta_log10_H"}


# ------------------------------------------------------------------ check sets

def item_rows(k, regime, variants=("base", "swap", "twin")):
    """One eval-type item's rows in the exporter's order."""
    return [row(f"t_{k:03d}_{v}", regime=regime, variant=v, pair=f"t_{k:03d}") for v in variants]


def test_select_rows_takes_whole_items_in_bank_order():
    rows = (item_rows(0, "near_only") + item_rows(1, "both") + item_rows(2, "none")
            + item_rows(3, "near_only", ("base", "swap")) + item_rows(4, "both", ("base",)))
    assert [r["uid"] for r in te.select_rows(rows, {})] == [r["uid"] for r in rows if r["regime"] != "none"]
    # 7 rows allowed: items 0, 1 (3 rows each) fit, item 3 (2 rows) would make 8, so the cut is before it
    assert [r["uid"] for r in te.select_rows(rows, {"rows_per_type": 7})] == [
        "t_000_base", "t_000_swap", "t_000_twin", "t_001_base", "t_001_swap", "t_001_twin"]
    picked = te.select_rows(rows, {"variants": ["base", "swap"], "regimes": ["both"]})
    assert [r["uid"] for r in picked] == ["t_001_base", "t_001_swap", "t_004_base"]


def test_arm_selected_by_name_family_or_method():
    arms = [{"arm": "caa", "family": "method"}, {"arm": "curve-snap", "family": "method"},
            {"arm": "curve-snap@0", "family": "control"}, {"arm": "isotropic[3]", "family": "isotropic"},
            {"arm": "shuffled:caa[1]", "family": "shuffled:caa"}, {"arm": "reward", "family": "control"}]
    pick = lambda wanted: [a["arm"] for a in arms if te.arm_selected(a, wanted)]
    assert pick(None) == [a["arm"] for a in arms]
    assert pick(["caa", "isotropic"]) == ["caa", "isotropic[3]"]
    assert pick(["curve-snap", "shuffled:caa"]) == ["curve-snap", "curve-snap@0", "shuffled:caa[1]"]
    assert pick(["control"]) == ["curve-snap@0", "reward"]


def test_load_set_refuses_what_it_cannot_run(tmp_path):
    def write(sel):
        (tmp_path / "s.json").write_text(json.dumps(sel))
        return tmp_path / "s.json"
    assert te.load_set(write({"types": {"toy": "expected to work"}, "rows_per_type": 4}))["rows_per_type"] == 4
    for bad, why in (({"types": {"toy": "x"}, "row_per_type": 4}, "unknown keys"), ({"rows_per_type": 4}, "types"),
                     ({"types": {"toy": "x"}, "regimes": ["none"]}, "no readout")):
        with pytest.raises(SystemExit, match=why):
            te.load_set(write(bad))


def test_baseline_runs_only_the_selected_rows(tmp_path):
    bank = tmp_path / "bank"
    write_bank(bank, item_rows(0, "near_only") + item_rows(1, "both") + item_rows(2, "near_only"))
    chat = FakeChat()
    te.stage_baseline(bank, tmp_path / "run", chat, SETTINGS, {"variants": ["base", "swap"], "rows_per_type": 4})
    assert chat.seen == ["prompt t_000_base", "prompt t_000_swap", "prompt t_001_base", "prompt t_001_swap"]
    with pytest.raises(SystemExit):                      # another selection is another run: refused in this run dir
        te.stage_baseline(bank, tmp_path / "run", chat, SETTINGS, {"rows_per_type": 4})


def test_main_with_a_check_set(tmp_path):
    """--set: its types only, outputs under --out/<set name>/, each type's expectation in the report."""
    t0, banks, out = make_runs(tmp_path)               # results under tmp_path/transfer/<type>
    (tmp_path / "t0run").mkdir()
    (tmp_path / "t0run" / "run.json").write_text(json.dumps({"settings": {**SETTINGS, "model": "toy/model"}}))
    sel = tmp_path / "transfer.json"
    sel.write_text(json.dumps({"description": "toy pilot", "types": {"toy": "expected to work"}}))
    common = ["--banks", str(banks), "--t0-bank", "b", "--t0-run", str(tmp_path / "t0run"), "--t0-out", str(t0),
              "--activations", "a", "--methods", "m", "--draws", "d", "--out", str(tmp_path)]
    te.main(["report", "--set", str(sel), *common])
    rep = json.loads((out / "transfer_report.json").read_text())
    assert set(rep["types"]) == {"toy"} and rep["set"]["name"] == "transfer"
    md = (out / "transfer_report.md").read_text()
    assert "| toy (expected to work) | prompts |" in md and "toy pilot" in md
    with pytest.raises(SystemExit, match="--types and --set"):
        te.main(["report", "--set", str(sel), "--types", "toy", *common])


PILOT = Path(__file__).parent / "transfer_sets" / "pilot_3types.json"
EXPORTED = Path.home() / "SPAR" / "results" / "horizon" / "evaltype_banks"


@pytest.mark.skipif(not all((EXPORTED / t / "prompts.jsonl").exists()
                            for t in json.loads(PILOT.read_text())["types"]), reason="eval-type banks not exported here")
def test_pilot_set_on_the_exported_banks():
    """The pilot is 300 prompts: per type 100 rows, 20 near and 20 both items, swaps for all and twins for 10 each."""
    sel = te.load_set(PILOT)
    for t in sel["types"]:
        rows = te.select_rows(rv.read_jsonl(EXPORTED / t / "prompts.jsonl"), sel)
        count = lambda regime, variant: sum(r["regime"] == regime and r["notes"]["variant"] == variant for r in rows)
        assert len(rows) == 100, t
        assert [count(g, v) for g in ("near_only", "both") for v in ("base", "swap", "twin")] == [20, 20, 10] * 2, t
