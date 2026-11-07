"""Running a hosted model (Claude, GPT, ...) through OpenRouter.

OpenRouter (https://openrouter.ai) serves many providers' models behind one OpenAI-compatible API, so a
single client covers e.g. `anthropic/claude-haiku-5.5` and `openai/gpt-5.6-luna`. The model is named by its
OpenRouter id; the list is at https://openrouter.ai/api/v1/models.

The API key is read from the environment variable OPENROUTER_API_KEY and never written to any output file.

Replies come back as the same `Reply` objects the local model produces (pbank/local_model.py), so callers do
not need to know which kind of model they are talking to. Differences worth knowing:

* `thinking` holds whatever reasoning text the provider returns. Some return none, or only a summary;
  hosted thinking is therefore *not* comparable to a local model's full trace.
* `n_generated` is the provider's count of completion tokens (it includes reasoning tokens where the
  provider bills them), so it is not comparable across tokenizers either.
* A request that fails after the client's retries is recorded with `error` set, rather than stopping the
  whole run; rerunning the same command retries only the failed prompts.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from .local_model import Reply

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterModel:
    """A hosted chat model on OpenRouter.

        model = OpenRouterModel("anthropic/claude-haiku-5.5", reasoning_effort="medium")
        replies = model.generate(["prompt 1", "prompt 2"], max_new_tokens=16000)
    """

    def __init__(self, model_id: str, reasoning_effort: Optional[str] = "medium", max_workers: int = 8):
        from openai import OpenAI

        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("set OPENROUTER_API_KEY to use a hosted model through OpenRouter")
        self.model_id = model_id
        self.reasoning_effort = reasoning_effort
        self.max_workers = max_workers
        # max_retries: the client retries rate limits, timeouts and 5xx errors with backoff on its own.
        self.client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key, max_retries=5)

    def _one(self, user_text: str, max_new_tokens: int, json_schema: Optional[dict]) -> Reply:
        import openai

        t0 = time.time()
        kwargs = {}
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if json_schema is not None:
            # Constrains the reply to valid JSON matching the schema, for providers that support it.
            kwargs["response_format"] = {"type": "json_schema",
                                         "json_schema": {"name": "result", "strict": True, "schema": json_schema}}
        try:
            resp = self.client.chat.completions.create(
                model=self.model_id, messages=[{"role": "user", "content": user_text}],
                max_tokens=max_new_tokens, **kwargs)
        except openai.APIError as e:   # base class of every error the client raises after its retries
            return Reply(thinking=None, answer=None, n_generated=0, hit_token_cap=False,
                         seconds=time.time() - t0, error=f"{type(e).__name__}: {e}")
        choice = resp.choices[0]
        # OpenRouter adds the reasoning text as a non-standard `reasoning` field on the message.
        reasoning = (choice.message.model_extra or {}).get("reasoning")
        return Reply(thinking=reasoning, answer=choice.message.content,
                     n_generated=resp.usage.completion_tokens if resp.usage else 0,
                     hit_token_cap=choice.finish_reason == "length", seconds=time.time() - t0)

    def generate(self, user_texts: list[str], max_new_tokens: int, json_schema: Optional[dict] = None) -> list[Reply]:
        """One reply per user text, requested concurrently; results are in the same order as `user_texts`."""
        with ThreadPoolExecutor(self.max_workers) as pool:
            return list(pool.map(lambda t: self._one(t, max_new_tokens, json_schema), user_texts))

    def describe(self) -> dict:
        import openai
        return {"model_id": self.model_id, "via": OPENROUTER_BASE_URL, "reasoning_effort": self.reasoning_effort,
                "openai_client": openai.__version__}
