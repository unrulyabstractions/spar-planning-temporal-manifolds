"""Eval-awareness prompt sets, keyword scan, probe states (tiny random Qwen3 and the real Qwen3-8B
tokenizer) and probe fit (synthetic states with a planted direction). No model weights."""

import argparse
import json
import sys
from collections import Counter

import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegressionCV
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from pipeline.s1_data_vectors.analysis import eval_keyword_scan as scan
from pipeline.s1_data_vectors.analysis import eval_probe_fit as fit
from pipeline.s1_data_vectors.experiments import exp24_eval_probe_states as states
from spar_horizon import eval_awareness as ea
from spar_horizon.config import Settings
from spar_horizon.eval_awareness import CUE_WORDS, KEYWORDS, TEST_AWARENESS, implicit_pairs, keyword_hits, training_pairs
from spar_horizon.steering import encode

MODEL = "Qwen/Qwen3-8B"


@pytest.fixture(scope="module")
def tok():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.padding_side = "left"
    return tokenizer


@pytest.fixture(scope="module")
def tiny(tok):
    torch.manual_seed(0)
    config = Qwen3Config(hidden_size=32, intermediate_size=64, num_hidden_layers=19, num_attention_heads=4,
                         num_key_value_heads=2, vocab_size=len(tok), head_dim=8)
    return Qwen3ForCausalLM(config).eval()


# --- prompt sets ----------------------------------------------------------------------------------

def test_training_pairs_are_150_requests_in_two_framings():
    records = training_pairs()
    assert len(records) == 300
    by_request = {}
    for r in records:
        by_request.setdefault(r["request_id"], {})[r["framing"]] = r
    assert sorted(by_request) == list(range(150))
    for framings in by_request.values():
        assert set(framings) == {"eval", "deploy"}
        eval_prompt, deploy_prompt = framings["eval"]["prompt"], framings["deploy"]["prompt"]
        assert eval_prompt[1] == deploy_prompt[1] and eval_prompt[1]["role"] == "user"
        assert eval_prompt[0]["role"] == deploy_prompt[0]["role"] == "system"
        assert eval_prompt[0]["content"] != deploy_prompt[0]["content"]
        assert framings["eval"]["pair_id"] == framings["deploy"]["pair_id"]
    assert len({r["prompt"][1]["content"] for r in records}) == 150


def test_each_framing_uses_every_paraphrase_and_none_goes_with_a_kind_of_request():
    records = training_pairs()
    for framing, paraphrases in (("eval", ea.EVAL_SYSTEM_MESSAGES), ("deploy", ea.DEPLOY_SYSTEM_MESSAGES)):
        used = Counter(r["prompt"][0]["content"] for r in records if r["framing"] == framing)
        assert set(used) == set(paraphrases)
        assert min(used.values()) >= 30
    templates_per_paraphrase = {}
    for r in records:
        if r["framing"] == "eval":
            templates_per_paraphrase.setdefault(r["prompt"][0]["content"], set()).add(r["request_id"] // 5)
    assert all(len(t) >= 10 for t in templates_per_paraphrase.values())


def test_the_requests_themselves_say_nothing_about_tests():
    assert not [r for r in ea.neutral_requests() if KEYWORDS.search(r) or CUE_WORDS.search(r)]


def test_implicit_pairs_are_60_requests_in_two_cues_with_no_test_words():
    records = implicit_pairs()
    assert len(records) == 120
    cues = {}
    for r in records:
        cues.setdefault(r["pair_id"], {})[r["cue"]] = r["prompt"]
    assert len(cues) == 60 and all(set(c) == {"benchmark", "natural"} for c in cues.values())
    assert not [r for r in records if KEYWORDS.search(r["prompt"]) or CUE_WORDS.search(r["prompt"])]
    assert all(r["prompt"].startswith("Question ") and r["prompt"].endswith("Answer with the letter only.")
               for r in records if r["cue"] == "benchmark")


# --- keyword pattern ---------------------------------------------------------------------------------

@pytest.mark.parametrize("text,matched", [
    ("Maybe the user is testing me.", "testing"),
    ("This looks like a Trick Question.", "Trick Question"),
    ("I should evaluate each option.", "evaluate"),
    ("It might be designed to check whether I notice.", "designed to check"),
    ("Perhaps I am being graded here.", "being graded"),
    ("A BENCHMARK for sure.", "BENCHMARK"),
])
def test_keywords_hit(text, matched):
    assert [m for m, _ in keyword_hits(text)] == [matched]


@pytest.mark.parametrize("text", [
    "The contest and the latest results.", "A protester attested to it.", "I will assess the options.",
    "Choose the fastest route.", "The score was three.",
])
def test_keywords_miss_on_substrings_and_other_words(text):
    assert keyword_hits(text) == []


@pytest.mark.parametrize("text", [
    "However, the problem might be testing the understanding of time value of money.",
    "Maybe the user is testing if I can recognize that.", "Perhaps it's a trick question.",
    "This is a test of reasoning.", "I am being graded.",
])
def test_strict_pattern_hits_the_prompt_called_a_test(text):
    assert keyword_hits(text, pattern=TEST_AWARENESS)


@pytest.mark.parametrize("text", [
    "The investment is being evaluated over five years.", "Let's assume a hypothetical rate of return.",
    "I should evaluate each option.", "After testing the product, it works.",
])
def test_strict_pattern_misses_weighing_options(text):
    assert keyword_hits(text, pattern=TEST_AWARENESS) == []


def test_a_hit_carries_its_context_on_one_line():
    (match, snippet), = keyword_hits("a" * 100 + "\nthe user is testing me\n" + "b" * 100, context=10)
    assert match == "testing" and "\n" not in snippet and "user is testing me" in snippet and len(snippet) < 40


# --- keyword scan on a fake results dir ---------------------------------------------------------------

def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def fake_results(root):
    write_json(root / "exp0_Qwen3-8B_think-on_full" / "answers.json", [
        {"think_text": "The user is testing me. Then evaluate it."}, {"think_text": "Just compute."},
        {"think_text": ""}])
    write_json(root / "evaltypes_Qwen3-8B_think-on" / "alpha" / "rows.json", [
        {"think_text": "A trick question?"}, {"think_text": "No."}, {"answer": None}])
    write_json(root / "related_Qwen3-8B_think-on" / "beta" / "rows.json", [{"think_text": "plain"}])
    write_json(root / "evaltypes_Qwen3-8B_think-on" / "summary.json", {"not": "a type"})


def test_scan_counts_traces_hits_and_words(tmp_path):
    fake_results(tmp_path)
    report = scan.scan(tmp_path)
    t0 = report["sources"]["t0_full"]
    assert (t0["n_traces"], t0["n_with_hit"], t0["share_with_hit"]) == (2, 1, 0.5)
    assert dict(t0["top_words"]) == {"testing": 1, "evaluate": 1}
    assert report["sources"]["evaltypes/alpha"]["n_traces"] == 2
    assert report["sources"]["related/beta"]["share_with_hit"] == 0
    assert sorted(report["sources"]) == ["evaltypes/alpha", "related/beta", "t0_full"]
    assert (t0["n_with_strict_hit"], t0["share_with_strict_hit"]) == (1, 0.5)
    assert report["sources"]["evaltypes/alpha"]["n_with_strict_hit"] == 1
    # snippets come from strict hits only: "user is testing", "trick question" (not "evaluate")
    assert sorted(s["match"] for s in report["snippets"]) == ["trick question", "user is testing"]


def test_scan_samples_at_most_40_snippets_reproducibly_and_writes_both_files(tmp_path, monkeypatch):
    write_json(tmp_path / "exp0_Qwen3-8B_think-on_full" / "answers.json", [{"think_text": "the problem is testing it"}] * 100)
    first, second = scan.scan(tmp_path, seed=1), scan.scan(tmp_path, seed=1)
    assert len(first["snippets"]) == 40 and first == second
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["scan", str(tmp_path), "--out", str(out)])
    scan.main()
    assert json.loads((out / "eval_keyword_scan.json").read_text())["sources"]["t0_full"]["n_traces"] == 100
    markdown = (out / "eval_keyword_scan.md").read_text()
    assert "true positive?" in markdown and "| t0_full | 100 | 100.0% | 100.0% |" in markdown


# --- exp24: the position read, on the real tokenizer and a tiny model ----------------------------------

def test_the_last_prompt_token_is_the_newline_after_assistant_as_in_the_t0_suffix(tok):
    cfg = Settings(model=MODEL, thinking="on")
    prompts = [training_pairs()[0]["prompt"], implicit_pairs()[1]["prompt"], "A plain user string."]
    for prompt in prompts:
        ids = encode(tok, prompt, cfg)
        assert tok.convert_ids_to_tokens(ids[-5:]) == ["<|im_end|>", "Ċ", "<|im_start|>", "assistant", "Ċ"]
        assert tok.decode(ids[-1]) == "\n"


def tiny_args(tmp_path, limit=None, layers="12,16"):
    return argparse.Namespace(model=MODEL, layers=layers, batch_size=4, limit=limit, out=str(tmp_path))


def test_states_are_the_last_token_state_in_every_batch_arrangement(tok, tiny):
    from spar_horizon.steering import forward
    cfg = Settings(model=MODEL, thinking="on")
    prompts = [r["prompt"] for r in training_pairs()[:5]] + [implicit_pairs()[0]["prompt"]]
    batched = states.last_token_states(tiny, "cpu", tok, cfg, prompts, (12, 16), batch_size=4)
    assert batched.shape == (6, 2, 32)
    for row, prompt in zip(batched, prompts):
        _, hidden, _ = forward(tiny, "cpu", [encode(tok, prompt, cfg)], tok.pad_token_id, hidden=True)
        expected = np.stack([hidden[layer][0, -1].numpy() for layer in (12, 16)])
        np.testing.assert_allclose(row, expected, atol=1e-5)


def test_main_writes_states_labels_and_texts(tok, tiny, tmp_path, monkeypatch):
    monkeypatch.setattr(states, "load_model", lambda cfg: (tok, tiny, "cpu"))
    states.main(tiny_args(tmp_path, limit=6))
    out_dir = tmp_path / "eval_probe_states_Qwen3-8B"
    z = np.load(out_dir / "states.npz")
    assert z["layers"].tolist() == [12, 16]
    assert z["train_X"].shape == (6, 2, 32) and z["implicit_X"].shape == (6, 2, 32)
    assert z["train_label"].tolist() == [1, 0, 1, 0, 1, 0] and z["train_request_id"].tolist() == [0, 0, 1, 1, 2, 2]
    assert z["implicit_label"].tolist() == [1, 0, 1, 0, 1, 0]
    rows = json.loads((out_dir / "rows.json").read_text())
    assert rows["train"][0]["text"].startswith("This conversation is part of an evaluation")
    assert rows["implicit"][1]["text"] == implicit_pairs()[1]["prompt"]


def test_layers_beyond_the_model_are_refused(tok, tiny):
    cfg = Settings(model=MODEL, thinking="on")
    with pytest.raises(ValueError, match="19 layers"):
        states.last_token_states(tiny, "cpu", tok, cfg, ["hi"], (19,), batch_size=1)


# --- probe fit on synthetic states -----------------------------------------------------------------------

D = 16


def synthetic_states(directory, implicit_shift=1.0, seed=0):
    """Layer 0 carries a planted direction for eval and for benchmark; layer 1 is noise. The training rows
    of one request share an offset, so splitting a pair across folds would leak."""
    rng = np.random.default_rng(seed)
    train, implicit = training_pairs(), implicit_pairs()
    y = np.array([r["framing"] == "eval" for r in train], dtype=int)
    request_id = np.array([r["request_id"] for r in train])
    implicit_y = np.array([r["cue"] == "benchmark" for r in implicit], dtype=int)
    direction = np.zeros(D)
    direction[0] = 1.0
    offset = rng.normal(size=(150, D))[request_id]
    train_X = np.stack([offset + rng.normal(size=(300, D)) + 3.0 * y[:, None] * direction,
                        offset + rng.normal(size=(300, D))], axis=1)
    implicit_X = np.stack([rng.normal(size=(120, D)) + implicit_shift * implicit_y[:, None] * direction,
                           rng.normal(size=(120, D))], axis=1)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / "states.npz", layers=np.array([22, 24]), train_X=train_X.astype(np.float32),
             implicit_X=implicit_X.astype(np.float32), train_label=y, train_request_id=request_id,
             implicit_label=implicit_y)
    rows = {"train": [{**r, "text": states.prompt_text(r["prompt"])} for r in train],
            "implicit": [{**r, "text": states.prompt_text(r["prompt"])} for r in implicit]}
    (directory / "rows.json").write_text(json.dumps(rows))
    return directory


@pytest.fixture(scope="module")
def planted(tmp_path_factory):
    directory = synthetic_states(tmp_path_factory.mktemp("states") / "eval_probe_states_Qwen3-8B")
    return directory, fit.fit_all(directory)


def test_the_planted_layer_wins_on_cv_and_passes_the_implicit_check(planted):
    _, probe = planted
    assert probe["layers"].tolist() == [22, 24]
    assert probe["cv_auroc"][0] > 0.9 and probe["cv_auroc"][1] < 0.7
    assert probe["best_layer"] == 22 and probe["passed"] is True
    assert probe["implicit_auroc"][0] > 0.7
    assert 0.0 <= probe["bow_implicit_auroc"] <= 1.0


def test_probe_arrays_have_the_documented_names_and_shapes(planted):
    _, probe = planted
    assert set(probe) == {"layers", "mean", "coef", "intercept", "cv_auroc", "implicit_auroc", "best_layer",
                          "passed", "bow_implicit_auroc"}
    assert probe["mean"].shape == probe["coef"].shape == (2, D)
    assert probe["intercept"].shape == probe["cv_auroc"].shape == probe["implicit_auroc"].shape == (2,)
    assert isinstance(probe["best_layer"], int) and isinstance(probe["passed"], bool)


def test_the_logit_formula_reproduces_sklearns_decision_function(planted):
    directory, probe = planted
    z = np.load(directory / "states.npz")
    X = z["train_X"][:, 0].astype(np.float64)
    folds = list(fit.GroupKFold(fit.N_FOLDS).split(X, z["train_label"], z["train_request_id"]))
    scale = X.std(0)
    model = LogisticRegressionCV(Cs=fit.C_GRID, cv=folds, scoring="roc_auc", max_iter=2000)
    model.fit((X - X.mean(0)) / scale, z["train_label"])
    ours = fit.probe_logit(X, probe["mean"][0], probe["coef"][0], probe["intercept"][0])
    np.testing.assert_allclose(ours, model.decision_function((X - X.mean(0)) / scale), atol=1e-4)


def test_cv_folds_keep_both_framings_of_a_request_together(planted, monkeypatch):
    directory, _ = planted
    z = np.load(directory / "states.npz")
    seen = []

    class Recording(fit.GroupKFold):
        def split(self, X, y=None, groups=None):
            for train, test in super().split(X, y, groups):
                seen.append((set(groups[train]), set(groups[test])))
                yield train, test

    monkeypatch.setattr(fit, "GroupKFold", Recording)
    fit.grouped_cv_auroc(z["train_X"][:, 0].astype(np.float64), z["train_label"], z["train_request_id"])
    assert len(seen) >= fit.N_FOLDS and all(not (a & b) for a, b in seen)


def test_an_unrelated_implicit_direction_fails_the_pass_rule(tmp_path):
    directory = synthetic_states(tmp_path / "eval_probe_states_Qwen3-8B", implicit_shift=0.0, seed=3)
    probe = fit.fit_all(directory)
    assert probe["best_layer"] == 22 and probe["passed"] is False and probe["implicit_auroc"][0] < 0.7


def test_main_writes_the_probe_file_and_scores_saved_runs(planted, tmp_path, monkeypatch):
    directory, probe = planted
    results = tmp_path / "results"
    rng = np.random.default_rng(0)

    def activations(n, **extra):
        return {"region_suffix": np.array([0, 5]), "region_prelabel": np.array([10, 11]),
                "L22": rng.normal(size=(n, 17, D)).astype(np.float16), **extra}

    t0_dir = results / "exp0_Qwen3-8B_think-on_full"
    t0_dir.mkdir(parents=True)
    np.savez(t0_dir / "activations.npz", **activations(4, horizons=np.array([1.0, np.nan, 2.0, 3.0])))
    type_dir = results / "evaltypes_Qwen3-8B_think-on" / "alpha"
    type_dir.mkdir(parents=True)
    np.savez(type_dir / "acts_0.npz", **activations(3, layers=np.array([22])))
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["fit", str(directory), "--out", str(out), "--score-runs", str(results)])
    fit.main()

    saved = np.load(out / "eval_probe_Qwen3-8B.npz")
    assert set(saved.files) == set(probe)
    report = json.loads((out / "eval_probe_report.json").read_text())
    assert set(report["sources"]) == {"t0_full", "evaltypes/alpha"}
    # layer 24 is not in the saved activations, so only layer 22 is scored
    assert list(report["sources"]["t0_full"]) == ["22"]
    assert report["sources"]["t0_full"]["22"]["suffix"]["n"] == 3
    suffix = np.load(t0_dir / "activations.npz")["L22"][[0, 2, 3]][:, 4].astype(np.float32)
    expected = fit.probe_logit(suffix, probe["mean"][0], probe["coef"][0], probe["intercept"][0]).mean()
    assert report["sources"]["t0_full"]["22"]["suffix"]["mean_logit"] == pytest.approx(expected, rel=1e-4)
    assert set(report["reference"]["22"]) == {"train_eval", "train_deploy", "implicit_benchmark", "implicit_natural"}
    assert report["reference"]["22"]["train_eval"]["mean_logit"] > report["reference"]["22"]["train_deploy"]["mean_logit"]
    markdown = (out / "eval_probe_report.md").read_text()
    assert "| t0_full | 3 |" in markdown and "reference: implicit_benchmark" in markdown
