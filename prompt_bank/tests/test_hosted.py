"""Hosted backends without network: the .env loader, checker specs, and the Anthropic request and reply mapping
(with a fake client standing in for the SDK's)."""

from types import SimpleNamespace

import pytest

from pbank.checker import CHECK_SCHEMA, parse_checker_spec, run_checker
from pbank.env import load_env_file
from pbank.records import read_jsonl
from pbank.t0 import build_verification_grid
from tests.test_checker import FakeHostedModel


def test_env_file_parsing_and_precedence(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# keys\nexport PBANK_TEST_A="quoted value"\nPBANK_TEST_B=plain\n\nPBANK_TEST_C=from-file\n')
    monkeypatch.delenv("PBANK_TEST_A", raising=False)
    monkeypatch.delenv("PBANK_TEST_B", raising=False)
    monkeypatch.setenv("PBANK_TEST_C", "from-environment")
    assert load_env_file((tmp_path / "missing.env", env)) == env
    import os
    assert os.environ["PBANK_TEST_A"] == "quoted value" and os.environ["PBANK_TEST_B"] == "plain"
    assert os.environ["PBANK_TEST_C"] == "from-environment"          # the environment wins over the file
    assert load_env_file((tmp_path / "missing.env",)) is None


def test_checker_specs():
    assert parse_checker_spec("anthropic:claude-haiku-5-5") == ("anthropic", "claude-haiku-5-5")
    assert parse_checker_spec("openrouter:anthropic/claude-haiku-5.5") == ("openrouter", "anthropic/claude-haiku-5.5")
    assert parse_checker_spec("local:Qwen/Qwen3.5-27B") == ("local", "Qwen/Qwen3.5-27B")
    for bad in ("claude-haiku-5-5", "anthropic:", "gpt:openai/x"):
        with pytest.raises(ValueError):
            parse_checker_spec(bad)


class FakeMessages:
    """Stands in for client.messages: records the request, returns a canned message (or raises)."""

    def __init__(self, response=None, exc=None):
        self.response, self.exc, self.requests = response, exc, []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if self.exc:
            raise self.exc
        return self.response


def anthropic_model(messages: FakeMessages):
    from pbank.hosted_model import AnthropicModel
    model = object.__new__(AnthropicModel)            # skip __init__: no key, no real client
    model.model_id, model.effort, model.max_workers = "claude-haiku-5-5", "medium", 2
    model.client = SimpleNamespace(messages=messages)
    return model


def message(stop_reason="end_turn", stop_details=None):
    return SimpleNamespace(
        content=[SimpleNamespace(type="thinking", thinking="Only a pays in time."),
                 SimpleNamespace(type="text", text='{"verdict": "ok"}')],
        stop_reason=stop_reason, stop_details=stop_details, usage=SimpleNamespace(output_tokens=321))


def test_anthropic_request_and_reply():
    pytest.importorskip("anthropic")
    fake = FakeMessages(message())
    [reply] = anthropic_model(fake).generate(["check this"], max_new_tokens=8192, json_schema=CHECK_SCHEMA)
    request = fake.requests[0]
    print("request:", {k: v for k, v in request.items() if k != "output_config"},
          "output_config keys:", list(request["output_config"]))
    assert request["model"] == "claude-haiku-5-5" and request["max_tokens"] == 8192
    assert request["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert request["output_config"] == {"effort": "medium", "format": {"type": "json_schema", "schema": CHECK_SCHEMA}}
    assert request["messages"] == [{"role": "user", "content": "check this"}]
    assert (reply.thinking, reply.answer, reply.n_generated, reply.hit_token_cap, reply.error) == \
           ("Only a pays in time.", '{"verdict": "ok"}', 321, False, None)
    [capped] = anthropic_model(FakeMessages(message("max_tokens"))).generate(["x"], max_new_tokens=10)
    assert capped.hit_token_cap


def test_anthropic_refusal_and_api_errors_are_recorded_not_raised():
    anthropic = pytest.importorskip("anthropic")
    httpx = pytest.importorskip("httpx2")             # the HTTP library of anthropic >= 1.0
    [refused] = anthropic_model(FakeMessages(message("refusal", SimpleNamespace(category="cyber")))).generate(
        ["x"], max_new_tokens=10)
    assert refused.answer is None and refused.error == "refusal (category: cyber)"
    down = anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))
    [failed] = anthropic_model(FakeMessages(exc=down)).generate(["x"], max_new_tokens=10)
    print("recorded error:", failed.error)
    assert failed.answer is None and failed.error.startswith("APIConnectionError")


def test_anthropic_checker_is_treated_as_hosted(tmp_path):
    records = build_verification_grid()[:3]
    out = tmp_path / "checks" / "anthropic__fake.jsonl"
    # FakeHostedModel asserts that it is handed the JSON schema, as every hosted checker must be.
    run_checker(records, "anthropic:fake", out, model=FakeHostedModel(records, fail_uid=None))
    rows = read_jsonl(out)
    assert len(rows) == 3 and all(r["agrees"] for r in rows)
