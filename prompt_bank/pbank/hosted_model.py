"""Running a hosted model (Claude, GPT, ...): through OpenRouter, or through Anthropic's API directly.

Two backends, chosen by the prefix of the checker spec (pbank/checker.py):

* `openrouter:<id>`: OpenRouter (https://openrouter.ai) serves many providers' models behind one
  OpenAI-compatible API, so one client covers e.g. `anthropic/claude-haiku-5.5` and `openai/gpt-5.6-luna`
  (ids at https://openrouter.ai/api/v1/models). Key: OPENROUTER_API_KEY.
* `anthropic:<id>`: Claude models through the Anthropic SDK, e.g. `claude-haiku-5-5` (note: Anthropic's ids
  use hyphens, OpenRouter's use a dot and a provider prefix). Key: ANTHROPIC_API_KEY.

Keys are read from the environment, or from a git-ignored `.env` file (pbank/env.py); they are never written
to any output file.

Replies come back as the same `Reply` objects the local model produces (pbank/local_model.py), so callers do
not need to know which kind of model they are talking to. Differences worth knowing:

* `thinking` holds whatever reasoning text the provider returns. Claude models return a *summary* of their
  reasoning (requested with display="summarized"), never the raw trace; other providers may return nothing.
  Hosted thinking is therefore *not* comparable to a local model's full trace.
* `n_generated` is the provider's count of output tokens (including reasoning tokens where the provider bills
  them), so it is not comparable across tokenizers either.
* A request that fails after the client's own retries, or that the model declines (Claude's "refusal" stop
  reason), is recorded with `error` set rather than stopping the whole run; rerunning the same command retries
  only those prompts.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from .env import load_env_file
from .local_model import Reply

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _require_key(name: str, backend: str) -> str:
    load_env_file()
    key = os.environ.get(name)
    if not key:
        raise RuntimeError(f"set {name} (in the environment or in a .env file) to use {backend}")
    return key


def _failed(t0: float, error: str) -> Reply:
    return Reply(thinking=None, answer=None, n_generated=0, hit_token_cap=False, seconds=time.time() - t0,
                 error=error)


class _HostedModel:
    """What both backends share: concurrent requests, one per prompt, results in prompt order."""

    max_workers: int = 8

    def _one(self, user_text: str, max_new_tokens: int, json_schema: Optional[dict]) -> Reply:
        raise NotImplementedError

    def generate(self, user_texts: list[str], max_new_tokens: int, json_schema: Optional[dict] = None) -> list[Reply]:
        """One reply per user text, requested concurrently; results are in the same order as `user_texts`.
        With `json_schema`, the provider is asked to constrain the reply to JSON matching it."""
        with ThreadPoolExecutor(self.max_workers) as pool:
            return list(pool.map(lambda t: self._one(t, max_new_tokens, json_schema), user_texts))


class OpenRouterModel(_HostedModel):
    """A hosted chat model on OpenRouter.

        model = OpenRouterModel("anthropic/claude-haiku-5.5", reasoning_effort="medium")
        replies = model.generate(["prompt 1", "prompt 2"], max_new_tokens=16000)
    """

    def __init__(self, model_id: str, reasoning_effort: Optional[str] = "medium", max_workers: int = 8):
        from openai import OpenAI

        key = _require_key("OPENROUTER_API_KEY", "a model through OpenRouter")
        self.model_id, self.reasoning_effort, self.max_workers = model_id, reasoning_effort, max_workers
        # max_retries: the client retries rate limits, timeouts and 5xx errors with backoff on its own.
        self.client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key, max_retries=5)

    def _one(self, user_text: str, max_new_tokens: int, json_schema: Optional[dict]) -> Reply:
        import openai

        t0 = time.time()
        kwargs = {}
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if json_schema is not None:
            kwargs["response_format"] = {"type": "json_schema",
                                         "json_schema": {"name": "result", "strict": True, "schema": json_schema}}
        try:
            resp = self.client.chat.completions.create(
                model=self.model_id, messages=[{"role": "user", "content": user_text}],
                max_tokens=max_new_tokens, **kwargs)
        except openai.APIError as e:   # base class of every error the client raises after its retries
            return _failed(t0, f"{type(e).__name__}: {e}")
        choice = resp.choices[0]
        # OpenRouter adds the reasoning text as a non-standard `reasoning` field on the message.
        reasoning = (choice.message.model_extra or {}).get("reasoning")
        return Reply(thinking=reasoning, answer=choice.message.content,
                     n_generated=resp.usage.completion_tokens if resp.usage else 0,
                     hit_token_cap=choice.finish_reason == "length", seconds=time.time() - t0)

    def describe(self) -> dict:
        import openai
        return {"model_id": self.model_id, "via": OPENROUTER_BASE_URL, "reasoning_effort": self.reasoning_effort,
                "openai_client": openai.__version__}


class AnthropicModel(_HostedModel):
    """A Claude model through the Anthropic API, with adaptive thinking at a fixed effort.

        model = AnthropicModel("claude-haiku-5-5", effort="medium")
        replies = model.generate(["prompt 1", "prompt 2"], max_new_tokens=16000)
    """

    def __init__(self, model_id: str, effort: Optional[str] = "medium", max_workers: int = 8):
        import anthropic

        _require_key("ANTHROPIC_API_KEY", "a Claude model through the Anthropic API")
        self.model_id, self.effort, self.max_workers = model_id, effort, max_workers
        # The client reads ANTHROPIC_API_KEY itself, and retries rate limits, timeouts and 5xx errors with backoff.
        self.client = anthropic.Anthropic(max_retries=5)

    def _one(self, user_text: str, max_new_tokens: int, json_schema: Optional[dict]) -> Reply:
        import anthropic

        t0 = time.time()
        output_config = {}
        if self.effort:
            output_config["effort"] = self.effort          # how much the model thinks: low | medium | high | ...
        if json_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": json_schema}
        try:
            msg = self.client.messages.create(
                model=self.model_id, max_tokens=max_new_tokens,
                messages=[{"role": "user", "content": user_text}],
                # Adaptive thinking: the model decides how much to think. "summarized" returns a readable summary
                # of the reasoning for reviewers (the default, "omitted", returns empty thinking text).
                thinking={"type": "adaptive", "display": "summarized"},
                **({"output_config": output_config} if output_config else {}))
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:   # after the client's retries
            return _failed(t0, f"{type(e).__name__}: {e}")
        if msg.stop_reason == "refusal":
            category = getattr(msg.stop_details, "category", None) if msg.stop_details else None
            return _failed(t0, f"refusal (category: {category})")
        thinking = "\n\n".join(b.thinking for b in msg.content if b.type == "thinking" and b.thinking) or None
        answer = "".join(b.text for b in msg.content if b.type == "text")
        return Reply(thinking=thinking, answer=answer, n_generated=msg.usage.output_tokens,
                     hit_token_cap=msg.stop_reason == "max_tokens", seconds=time.time() - t0)

    def describe(self) -> dict:
        import anthropic
        return {"model_id": self.model_id, "via": "Anthropic API", "effort": self.effort,
                "thinking": "adaptive, summarized", "anthropic_client": anthropic.__version__}
