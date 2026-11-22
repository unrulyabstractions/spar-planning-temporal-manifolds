"""Finding the answer-label token, on the real Gemma 4 and Qwen3.5 tokenizers (skipped if not cached).

Replies are built the way a model's generated ids look: thinking, the end-of-thinking token, then the answer.
"""

import pytest

from steer.sites import LabelSite, locate_label, replay_sites


def tokenizer(model_id):
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    except OSError:
        pytest.skip(f"{model_id} tokenizer not cached")


FAMILIES = {   # model id -> (thinking prefix tokens, end-of-thinking token)
    "Qwen/Qwen3.5-2B": ("", "</think>"),
    "google/gemma-4-31B-it": ("<|channel>", "<channel|>"),
}


def reply_ids(tok, model_id, thinking_text, answer):
    opener, eot = FAMILIES[model_id]
    ids = [tok.convert_tokens_to_ids(opener)] if opener else []
    ids += tok.encode(("thought\n" if opener else "") + thinking_text, add_special_tokens=False)
    ids += [tok.convert_tokens_to_ids(eot)]
    return ids + tok.encode(answer, add_special_tokens=False)


CASES = [   # (answer, labels, expected label char)
    ("\n\nI choose: b). My reasoning: only b pays in time.", ("a)", "b)"), "b"),
    ("I choose: **2)**. My reasoning: both pay.", ("1)", "2)"), "2"),
    ("I choose: (a). My reasoning: it is sooner.", ("a)", "b)"), "a"),
    ("I choose: and then... I choose: a). My reasoning: x.", ("a)", "b)"), "a"),
]


@pytest.mark.parametrize("model_id", list(FAMILIES))
@pytest.mark.parametrize("answer,labels,char", CASES)
def test_label_token_found(model_id, answer, labels, char):
    tok = tokenizer(model_id)
    # The thinking mentions a different choice: it must not be taken for the answer.
    ids = reply_ids(tok, model_id, "Maybe I choose: a) here? No, think again.", answer)
    site = locate_label(tok, ids, labels, thinking=True)
    token = tok.decode([ids[site.label_index]])
    before = tok.decode(ids[site.answer_start:site.label_index])
    print(f"{model_id}: {answer[:28]!r} -> label token {token!r}, preceded by {before[-14:]!r}")
    assert site.label_char == char
    assert char in token
    assert char not in before.split("choose")[-1]          # nothing between "I choose" and the label carries it


@pytest.mark.parametrize("model_id", list(FAMILIES))
def test_no_site_without_answer_or_choice(model_id):
    tok = tokenizer(model_id)
    # Thinking never closed (cut off at the token cap): no answer, even though the thinking says "I choose: a)".
    no_eot = tok.encode("thinking and I choose: a) but never closing", add_special_tokens=False)
    assert locate_label(tok, no_eot, ("a)", "b)"), thinking=True) is None
    no_choice = reply_ids(tok, model_id, "hmm", "Neither option pays in time.")
    assert locate_label(tok, no_choice, ("a)", "b)"), thinking=True) is None
    wrong_labels = reply_ids(tok, model_id, "hmm", "I choose: c).")
    assert locate_label(tok, wrong_labels, ("a)", "b)"), thinking=True) is None


def test_thinking_off():
    tok = tokenizer("Qwen/Qwen3.5-2B")
    ids = tok.encode("I choose: 1). My reasoning: sooner.", add_special_tokens=False)
    site = locate_label(tok, ids, ("1)", "2)"), thinking=False)
    print(f"thinking off: label token {tok.decode([ids[site.label_index]])!r} at {site.label_index}")
    assert site.answer_start == 0 and "1" in tok.decode([ids[site.label_index]])


# ---- the horizon site ------------------------------------------------------------------------------------------------

def bank_records():
    from fixtures import BANK
    from pbank.records import load_prompts
    return load_prompts(BANK)


def chat_text(tok, text, thinking):
    return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=thinking)


@pytest.mark.parametrize("model_id", list(FAMILIES))
@pytest.mark.parametrize("thinking", [True, False])
def test_horizon_tokens_decode_back_for_every_prompt(model_id, thinking):
    from collections import Counter
    from steer.sites import locate_horizon
    tok = tokenizer(model_id)
    sizes, shifted_ok, first = Counter(), 0, set()
    recs = [r for r in bank_records() if r.horizon is not None]
    for r in recs:
        chat = chat_text(tok, r.text, thinking)
        site = locate_horizon(tok, chat, r.text, r.horizon)
        ids = tok(chat, add_special_tokens=False)["input_ids"]
        assert site.decoded.strip() == r.horizon, (r.uid, r.horizon, site.tokens)
        assert [ids[k] for k in site.token_indices] == site.token_ids and site.n_input_tokens == len(ids)
        assert r.text[site.char_start:site.char_end] == r.horizon
        assert r.text.rfind("CONSTRAINT: ", 0, site.char_start) > r.text.find("\n")     # in the CONSTRAINT line
        # Negative control: the same number of tokens one position later does not spell the horizon.
        shifted_ok += tok.decode([ids[k + 1] for k in site.token_indices]).strip() == r.horizon
        sizes[len(site.tokens)] += 1
        first.add(site.token_indices[0])
    twice = [r for r in recs if r.text.count(r.horizon) > 1]
    print(f"{model_id}, thinking {thinking}: {len(recs)} horizons decode back; tokens per horizon {dict(sorted(sizes.items()))}; "
          f"first token index from {min(first)} to {max(first)}; shifted by one decodes back for {shifted_ok}; "
          f"{len(twice)} prompts also write the horizon's words in an option ({[r.uid for r in twice]})")
    assert len(recs) == 671 and shifted_ok == 0 and len(twice) == 4


def test_horizon_span_refuses_ambiguous_text():
    from steer.sites import horizon_char_span
    text = "TASK: a) 1 dollar in 3 months.\nCONSTRAINT: Your planning horizon is 3 months. Choose.\nACTION: x"
    start, end = horizon_char_span(text, "3 months")
    assert text[start:end] == "3 months" and start > text.index("CONSTRAINT")
    for bad, why in ((text.replace("CONSTRAINT: ", "LIMIT: "), "no CONSTRAINT line"),
                     (text + "\nCONSTRAINT: again 3 months", "two CONSTRAINT lines"),
                     (text.replace("Choose.", "Not 3 months more."), "stated twice in the line")):
        with pytest.raises(ValueError) as e:
            horizon_char_span(bad, "3 months")
        print(f"{why}: {e.value}")


@pytest.mark.parametrize("model_id", list(FAMILIES))
def test_replay_sites_on_a_thinking_reply(model_id):
    """The four extraction sites of a replayed sequence = prompt ids + generated ids up to the label."""
    tok = tokenizer(model_id)
    prompt_ids = tok.encode("PROMPT: pick one.\n", add_special_tokens=False)
    gen = reply_ids(tok, model_id, "Compare the two. I choose: a) maybe.", "\n\nI choose: b). My reasoning: x.")
    site = locate_label(tok, gen, ("a)", "b)"), thinking=True)
    ids = prompt_ids + gen[:site.label_index + 1]
    where = replay_sites(len(prompt_ids), site, thinking=True)
    shown = {name: tok.decode([ids[k]]) for name, k in where.items()}
    print(f"{model_id}: {len(prompt_ids)} prompt + {site.label_index + 1} generated tokens; sites {where}, tokens {shown}")
    assert where["label"] == len(ids) - 1 and where["prelabel"] == len(ids) - 2
    assert ids[where["prompt_end"]] == prompt_ids[-1] and where["prompt_end"] + 1 == len(prompt_ids)
    assert shown["think_end"] == FAMILIES[model_id][1]                  # the end-of-thinking token itself
    assert "b" in shown["label"] and "choose" in tok.decode(ids[where["think_end"] + 1:where["prelabel"] + 1])


def test_replay_sites_thinking_off_and_refusals():
    first = replay_sites(10, LabelSite(label_index=0, answer_start=0, label_char="a"), thinking=False)
    later = replay_sites(10, LabelSite(label_index=3, answer_start=0, label_char="a"), thinking=False)
    print(f"label as the first generated token: {first}; as the fourth: {later}")
    assert first == {"label": 10, "prelabel": 9, "prompt_end": 9}      # prelabel is then the prompt's last token
    assert later == {"label": 13, "prelabel": 12, "prompt_end": 9} and "think_end" not in later
    with pytest.raises(ValueError, match="end-of-thinking"):
        replay_sites(10, LabelSite(label_index=3, answer_start=0, label_char="a"), thinking=True)
    with pytest.raises(ValueError, match="out of order"):                # the label inside the thinking
        replay_sites(10, LabelSite(label_index=2, answer_start=5, label_char="a"), thinking=True)
