"""Running a local Hugging Face chat model: batched greedy generation, with the thinking split off the answer.

Used by both the baseline run (baseline.py) and the local prompt checker (pbank/checker.py), so the two see
models through exactly the same code.

How a reply is split. In thinking mode a model first writes its reasoning and then closes it with an
end-of-thinking token: `</think>` for Qwen3 / Qwen3.5, `<channel|>` for Gemma 4 (the end of its "thought"
channel). Everything before the FIRST such token is the thinking; everything after it is the answer. If the
model never closes its thinking (it hit the token cap, or wrote something malformed) the answer is None: we do
not read a choice out of unfinished deliberation. The end-of-thinking token is looked up in the tokenizer, and
a model family with none of the known tokens is refused rather than guessed (tests/test_local_model.py checks
the templates of the families in use).

Settings fixed here on purpose: greedy decoding (do_sample=False), so a run is repeatable on the same
hardware and batch composition; bf16 weights; left padding so every sequence in a batch ends at the same index.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

END_OF_THINKING_TOKENS = ("</think>", "<channel|>")   # Qwen3 / Qwen3.5, Gemma 4
END_OF_TURN_TOKENS = ("<|im_end|>", "<turn|>")         # Qwen3 / Qwen3.5, Gemma 4
GEMMA_CHANNEL_OPEN = "<|channel>"                      # Gemma 4: opens a channel, e.g. "<|channel>thought\n"


@dataclass
class Reply:
    """One model reply, split into thinking and answer."""
    thinking: Optional[str]     # None in non-thinking mode
    answer: Optional[str]       # None if the model never closed its thinking
    n_generated: int            # tokens generated, including thinking
    hit_token_cap: bool         # stopped at max_new_tokens rather than ending its turn
    seconds: float              # wall time of the batch this reply was generated in, divided by batch size
    error: Optional[str] = None # set when the request itself failed (hosted models); the other fields are empty


def _single_token_id(tok, candidates: tuple[str, ...], what: str) -> int:
    """The id of whichever of `candidates` the tokenizer knows. Exactly one must be known."""
    found = {}
    for t in candidates:
        i = tok.convert_tokens_to_ids(t)
        if i is not None and i != tok.unk_token_id:
            found[t] = i
    if len(found) != 1:
        raise ValueError(f"expected exactly one {what} token among {candidates}, the tokenizer has {sorted(found)}. "
                         f"This model family's chat template has not been verified for pbank.")
    return next(iter(found.values()))


def split_reply(tok, generated_ids: list[int], thinking: bool) -> tuple[Optional[str], Optional[str]]:
    """(thinking text, answer text) from the generated token ids; see the module docstring."""
    if not thinking:
        return None, tok.decode(generated_ids, skip_special_tokens=True).strip()
    eot = _single_token_id(tok, END_OF_THINKING_TOKENS, "end-of-thinking")
    if eot not in generated_ids:
        return tok.decode(generated_ids, skip_special_tokens=True).strip(), None
    k = generated_ids.index(eot)
    thought = tok.decode(generated_ids[:k], skip_special_tokens=True).strip()
    # Gemma 4 opens its thinking with "<|channel>thought\n". decode() drops the special token but not the
    # channel name, so remove it, so that thinking text reads the same across model families.
    if generated_ids and generated_ids[0] == tok.convert_tokens_to_ids(GEMMA_CHANNEL_OPEN):
        thought = thought.removeprefix("thought").lstrip("\n")
    return thought, tok.decode(generated_ids[k + 1:], skip_special_tokens=True).strip()


class LocalChatModel:
    """A Hugging Face causal LM behind its own chat template, generating greedily in batches.

        model = LocalChatModel("google/gemma-4-31B-it")
        replies = model.generate(["prompt 1", "prompt 2"], thinking=True, max_new_tokens=8192)
    """

    def __init__(self, model_id: str, dtype: str = "bfloat16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype), device_map="auto")
        self.model.eval()
        off_gpu = sorted({str(p.device) for p in self.model.parameters() if p.device.type != "cuda"})
        if off_gpu:
            raise RuntimeError(f"{model_id} did not fit on the GPUs (some weights are on {off_gpu}); "
                               f"offloaded generation would be far too slow for a baseline run")
        self.end_ids = sorted({_single_token_id(self.tok, END_OF_TURN_TOKENS, "end-of-turn"), self.tok.eos_token_id})
        self.input_device = self.model.get_input_embeddings().weight.device

    def chat_text(self, user_text: str, thinking: bool) -> str:
        """The exact string the model is given: its chat template applied to one user turn."""
        return self.tok.apply_chat_template([{"role": "user", "content": user_text}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=thinking)

    def generate(self, user_texts: list[str], thinking: bool, max_new_tokens: int) -> list[Reply]:
        """One reply per user text, generated together as one batch."""
        import torch

        t0 = time.time()
        chats = [self.chat_text(t, thinking) for t in user_texts]
        # The chat template already contains the special tokens (e.g. <bos>); don't add them twice.
        batch = self.tok(chats, return_tensors="pt", padding=True, add_special_tokens=False).to(self.input_device)
        with torch.no_grad():
            out = self.model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False,
                                      eos_token_id=self.end_ids, pad_token_id=self.tok.pad_token_id)
        seconds = (time.time() - t0) / len(user_texts)

        replies = []
        prompt_len = batch["input_ids"].shape[1]
        for row in out[:, prompt_len:].tolist():
            # Cut at the first end-of-turn token; anything after it is padding.
            end = next((k for k, t in enumerate(row) if t in self.end_ids), None)
            ids = row if end is None else row[:end]
            thought, answer = split_reply(self.tok, ids, thinking)
            replies.append(Reply(thinking=thought, answer=answer, n_generated=len(ids),
                                 hit_token_cap=end is None, seconds=seconds))
        return replies

    def describe(self) -> dict:
        """What to record about the model and software in a run's metadata."""
        import importlib.metadata
        import torch
        import transformers

        def version(package: str):
            try:
                return importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                return None

        return {
            "model_id": self.model_id,
            "model_revision": model_revision(self.model_id),
            "dtype": str(self.model.dtype),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            # Qwen3.5's linear-attention layers run on these kernels when installed, else on a reference
            # implementation; the two are each deterministic but not bit-identical, so record which ran.
            "flash_linear_attention": version("flash-linear-attention"),
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        }


def model_revision(model_id: str) -> Optional[str]:
    """The commit of the weights in use: the Hugging Face cache keeps each download under
    .../snapshots/<commit hash>/, so read it from the path of the cached config.json. None if not cached
    (e.g. a local directory was given instead of a hub id)."""
    from huggingface_hub import try_to_load_from_cache
    path = try_to_load_from_cache(model_id, "config.json")
    return Path(path).parent.name if isinstance(path, str) else None


def log(msg: str) -> None:
    """Print a progress line and flush at once, so a run's log survives if the process is killed."""
    print(msg, flush=True)
    sys.stdout.flush()
