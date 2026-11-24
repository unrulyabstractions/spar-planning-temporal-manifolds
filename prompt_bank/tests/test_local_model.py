"""Splitting a generation into thinking and answer, with the real tokenizers of the model families in use.

Tokenizer-only (no weights, no GPU). Each test is skipped if that tokenizer is not in the local Hugging Face
cache; run with HF_HUB_OFFLINE=1 to make sure nothing is downloaded.
"""

import pytest

from pbank.local_model import split_reply


def tokenizer(model_id):
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(model_id)
    except OSError:
        pytest.skip(f"{model_id} tokenizer not cached")


def ids(tok, text):
    return tok(text, add_special_tokens=False)["input_ids"]


def test_gemma4_thinking_template_and_split():
    tok = tokenizer("google/gemma-4-31B-it")
    chat = tok.apply_chat_template([{"role": "user", "content": "Q"}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=True)
    print(repr(chat))
    assert "<|think|>" in chat and chat.endswith("<|turn>model\n")     # thinking on; the model opens its channel
    generated = ids(tok, "<|channel>thought\nOnly a pays within 8 years.<channel|>I choose: a). My reasoning: x")
    thought, answer = split_reply(tok, generated, thinking=True)
    print(f"thinking={thought!r}\nanswer={answer!r}")
    assert thought == "Only a pays within 8 years." and answer == "I choose: a). My reasoning: x"
    # Never closed: all of it is thinking and there is no answer.
    assert split_reply(tok, ids(tok, "<|channel>thought\nStill thinking about it"), thinking=True)[1] is None


@pytest.mark.parametrize("model_id", ["Qwen/Qwen3.5-27B", "Qwen/Qwen3.5-2B"])
def test_qwen35_thinking_template_and_split(model_id):
    tok = tokenizer(model_id)
    chat = tok.apply_chat_template([{"role": "user", "content": "Q"}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=True)
    print(repr(chat))
    assert chat.endswith("<think>\n")                                 # the template opens the thinking block
    generated = ids(tok, "Both pay; b is larger.\n</think>\n\nI choose: b). My reasoning: y")
    thought, answer = split_reply(tok, generated, thinking=True)
    print(f"thinking={thought!r}\nanswer={answer!r}")
    assert thought == "Both pay; b is larger." and answer == "I choose: b). My reasoning: y"
    # A thought that happens to start with the word "thought" is not trimmed (that is Gemma-only).
    assert split_reply(tok, ids(tok, "thought about it</think>ok"), thinking=True) == ("thought about it", "ok")


def test_unverified_family_is_refused():
    tok = tokenizer("gpt2")
    with pytest.raises(ValueError, match="not been verified"):
        split_reply(tok, ids(tok, "hello"), thinking=True)


# ---- decoding: greedy or seeded sampling (no weights needed) ---------------------------------------------------------

def test_auto_decoding_samples_only_when_thinking_is_on():
    from pbank.local_model import GREEDY, resolve_decoding
    gemma = {"temperature": 1.0, "top_p": 0.95, "top_k": 64}       # as in Gemma 4's generation_config.json
    on = resolve_decoding("auto", True, "m", model_defaults=gemma)
    off = resolve_decoding("auto", False, "m", model_defaults=gemma)
    forced = resolve_decoding("sample", False, "m", temperature=0.6, top_k=20, seed=3, model_defaults=gemma)
    print(f"auto, thinking on: {on.settings()}\nauto, thinking off: {off.settings()}\nsample with options: {forced.settings()}")
    assert off is GREEDY and off.settings() == "greedy"            # the string greedy runs have always recorded
    assert resolve_decoding("greedy", True, "m", model_defaults=gemma) is GREEDY
    assert on.settings() == {"mode": "sample", "temperature": 1.0, "top_p": 0.95, "top_k": 64, "seed": 0,
                             "source": {k: "model generation_config" for k in ("temperature", "top_p", "top_k")}}
    assert (forced.temperature, forced.top_p, forced.top_k, forced.seed) == (0.6, 0.95, 20, 3)
    assert forced.source == {"temperature": "option", "top_p": "model generation_config", "top_k": "option"}


def test_sampling_values_are_never_guessed():
    from pbank.local_model import Decoding, resolve_decoding
    with pytest.raises(ValueError, match=r"--top-p --top-k") as e:          # a model with no generation config
        resolve_decoding("auto", True, "Qwen/Qwen3.5-2B", temperature=0.6, model_defaults={})
    print(f"missing values: {e.value}")
    with pytest.raises(ValueError, match="only apply to sampling"):         # sampling options on a greedy run
        resolve_decoding("auto", False, "m", temperature=0.6, model_defaults={})
    with pytest.raises(ValueError):
        Decoding("sample", 1.0, 0.95, None, 0)
    with pytest.raises(ValueError):
        Decoding("beam")


def test_batch_seed_depends_on_the_seed_and_the_batch_only():
    from pbank.local_model import Decoding
    d = Decoding("sample", 1.0, 0.95, 64, 0)
    a, b = d.batch_seed(["p1", "p2"]), d.batch_seed(["p1", "p2"])
    others = {d.batch_seed(["p2", "p1"]), d.batch_seed(["p1"]), d.batch_seed(["p1p2"]), d.batch_seed(["p1", "p2", ""]),
              Decoding("sample", 1.0, 0.95, 64, 1).batch_seed(["p1", "p2"])}
    print(f"batch seed {a}; 5 variations (order, subset, joined, extra empty prompt, other run seed) give {len(others)} other seeds")
    assert a == b == 3809895313129854352        # a fixed value: stable across processes (no Python hash() involved)
    assert a not in others and len(others) == 5 and 0 <= a < 2**63


@pytest.mark.parametrize("model_id,expected", [("google/gemma-4-31B-it", {"temperature": 1.0, "top_p": 0.95, "top_k": 64}),
                                               ("Qwen/Qwen3.5-2B", {})])
def test_model_sampling_defaults_read_the_models_own_config(model_id, expected):
    pytest.importorskip("transformers")
    from huggingface_hub import try_to_load_from_cache
    if not isinstance(try_to_load_from_cache(model_id, "config.json"), str):
        pytest.skip(f"{model_id} not cached")
    from pbank.local_model import model_sampling_defaults
    found = model_sampling_defaults(model_id)
    print(f"{model_id}: {found or 'no generation config'}")
    assert found == expected
