"""Running a local Hugging Face chat model: batched generation (greedy or seeded sampling), with the thinking split
off the answer.

Used by both the baseline run (baseline.py) and the local prompt checker (pbank/checker.py), so the two see
models through exactly the same code.

How a reply is split. In thinking mode a model first writes its reasoning and then closes it with an
end-of-thinking token: `</think>` for Qwen3 / Qwen3.5, `<channel|>` for Gemma 4 (the end of its "thought"
channel). Everything before the FIRST such token is the thinking; everything after it is the answer. If the
model never closes its thinking (it hit the token cap, or wrote something malformed) the answer is None: we do
not read a choice out of unfinished deliberation. The end-of-thinking token is looked up in the tokenizer, and
a model family with none of the known tokens is refused rather than guessed (tests/test_local_model.py checks
the templates of the families in use).

Settings fixed here on purpose: bf16 weights; left padding so every sequence in a batch ends at the same index.

Decoding (`Decoding`, passed to `generate`; greedy if not given):
  greedy   the most likely token at every step (do_sample=False). Repeatable on the same hardware and batch
           composition. With thinking on, small models often loop under it until the token cap.
  sample   sampling with a temperature, top-p and top-k, seeded. The random generator is re-seeded before every batch
           from the run's seed and the batch's own prompts, so the same batch with the same seed gives the same
           replies (on the same hardware and software), whatever was generated before it. A reply therefore depends on
           the seed and on which prompts share its batch, as a greedy reply already depends on its batch through
           padding. Nothing is taken from the model's generation config implicitly: the three values are always
           passed, and `resolve_decoding` is where a model's own recommended values are read and recorded.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import hashlib

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
    generated_ids: Optional[list[int]] = None   # local models: the exact generated token ids, up to (not including)
                                                # the end-of-turn token; lets a later pass replay the reply exactly


def single_token_id(tok, candidates: tuple[str, ...], what: str) -> int:
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
    eot = single_token_id(tok, END_OF_THINKING_TOKENS, "end-of-thinking")
    if eot not in generated_ids:
        return tok.decode(generated_ids, skip_special_tokens=True).strip(), None
    k = generated_ids.index(eot)
    thought = tok.decode(generated_ids[:k], skip_special_tokens=True).strip()
    # Gemma 4 opens its thinking with "<|channel>thought\n". decode() drops the special token but not the
    # channel name, so remove it, so that thinking text reads the same across model families.
    if generated_ids and generated_ids[0] == tok.convert_tokens_to_ids(GEMMA_CHANNEL_OPEN):
        thought = thought.removeprefix("thought").lstrip("\n")
    return thought, tok.decode(generated_ids[k + 1:], skip_special_tokens=True).strip()


@dataclass(frozen=True)
class Decoding:
    """How tokens are chosen. Greedy needs nothing else; sampling needs all four values."""
    mode: str = "greedy"                    # "greedy" | "sample"
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    seed: Optional[int] = None
    source: Optional[dict] = None           # sampling: where each value came from ("option" | "model generation_config")

    def __post_init__(self):
        if self.mode not in ("greedy", "sample"):
            raise ValueError(f"decoding mode must be greedy or sample, not {self.mode!r}")
        if self.mode == "sample" and None in (self.temperature, self.top_p, self.top_k, self.seed):
            raise ValueError(f"sampling needs temperature, top_p, top_k and seed: {self}")

    def settings(self):
        """What a run records, and what a resumed run must match: "greedy" (as runs before sampling existed
        recorded it), or the sampling values."""
        if self.mode == "greedy":
            return "greedy"
        return {"mode": "sample", "temperature": self.temperature, "top_p": self.top_p, "top_k": self.top_k,
                "seed": self.seed, "source": self.source}

    def batch_seed(self, texts: list[str]) -> int:
        """The seed for one batch: from the run's seed and the batch's prompts, in order."""
        h = hashlib.sha256(f"{self.seed}".encode())
        for text in texts:
            h.update(b"\x00" + text.encode())
        return int.from_bytes(h.digest()[:8], "big") >> 1      # 63 bits: accepted by torch.manual_seed


GREEDY = Decoding()


def model_sampling_defaults(model_id: str) -> dict:
    """temperature / top_p / top_k from the model's own generation_config.json: what its authors recommend. Empty
    if the model ships none (Qwen3.5), and a value the file does not set is left out."""
    from transformers import GenerationConfig
    try:
        config = GenerationConfig.from_pretrained(model_id)
    except OSError:
        return {}
    found = {k: getattr(config, k, None) for k in ("temperature", "top_p", "top_k")}
    return {k: v for k, v in found.items() if v is not None}


def add_decoding_args(parser, default: str = "auto") -> None:
    parser.add_argument("--decoding", default=default, choices=["auto", "greedy", "sample"],
                        help="greedy, or seeded sampling; auto (default: %(default)s) samples when thinking is on, "
                             "where greedy decoding tends to loop, and is greedy when it is off")
    parser.add_argument("--temperature", type=float, help="sampling: default from the model's generation config")
    parser.add_argument("--top-p", type=float, help="sampling: default from the model's generation config")
    parser.add_argument("--top-k", type=int, help="sampling: default from the model's generation config")
    parser.add_argument("--seed", type=int, default=0, help="sampling: the run's seed (default 0)")


def resolve_decoding(mode: str, thinking: bool, model_id: str, temperature: Optional[float] = None,
                     top_p: Optional[float] = None, top_k: Optional[int] = None, seed: int = 0,
                     model_defaults: Optional[dict] = None) -> Decoding:
    """The Decoding for a run: `mode` "auto" samples when thinking is on; sampling values come from the options given,
    else from the model's own generation config (`model_defaults`, read from the model if None). A value found in
    neither is an error, not a guess."""
    if mode == "greedy" or (mode == "auto" and not thinking):
        given = {k: v for k, v in (("--temperature", temperature), ("--top-p", top_p), ("--top-k", top_k)) if v is not None}
        if given:
            raise ValueError(f"{sorted(given)} only apply to sampling, and this run decodes greedily "
                             f"({'--decoding greedy' if mode == 'greedy' else 'thinking is off'}); add --decoding sample")
        return GREEDY
    defaults = model_sampling_defaults(model_id) if model_defaults is None else model_defaults
    values, source = {}, {}
    for name, option in (("temperature", temperature), ("top_p", top_p), ("top_k", top_k)):
        if option is not None:
            values[name], source[name] = option, "option"
        elif name in defaults:
            values[name], source[name] = defaults[name], "model generation_config"
    missing = [n for n in ("temperature", "top_p", "top_k") if n not in values]
    if missing:
        raise ValueError(f"sampling needs {missing}: {model_id} has no generation config that sets them, so give "
                         f"--{' --'.join(m.replace('_', '-') for m in missing)} (its model card's recommended values)")
    return Decoding("sample", float(values["temperature"]), float(values["top_p"]), int(values["top_k"]), seed, source)


def decoding_from_args(args, thinking: bool, model_id: str) -> Decoding:
    try:
        return resolve_decoding(args.decoding, thinking, model_id, args.temperature, args.top_p, args.top_k, args.seed)
    except ValueError as e:
        sys.exit(str(e))


class LocalChatModel:
    """A Hugging Face causal LM behind its own chat template, generating in batches (greedily unless told to sample).

        model = LocalChatModel("google/gemma-4-31B-it")
        replies = model.generate(["prompt 1", "prompt 2"], thinking=True, max_new_tokens=8192)
    """

    def __init__(self, model_id: str, dtype: str = "bfloat16", allow_offload: bool = False,
                 offload_headroom_gib: float = 4.0):
        """allow_offload: let weights that don't fit on the GPUs stay in CPU RAM (accelerate moves each offloaded
        block to the GPU to run it, so the arithmetic is unchanged, only far slower). For smoke tests of a model too
        big for the local GPUs; refused by default, because a baseline run offloaded would take days. With it, each
        GPU keeps `offload_headroom_gib` free of weights for the key-value cache and activations: left to itself,
        device_map="auto" fills the GPUs to ~150 MB free, and generation then runs out of memory as the cache grows
        (Gemma 4 31B on 2 x 16 GB, 2026-11-14)."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"
        max_memory = None
        if allow_offload:
            import os
            ram_gib = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
            max_memory = {i: f"{torch.cuda.get_device_properties(i).total_memory / 2**30 - offload_headroom_gib:.1f}GiB"
                          for i in range(torch.cuda.device_count())} | {"cpu": f"{0.85 * ram_gib:.0f}GiB"}
        self.max_memory = max_memory
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype), device_map="auto",
                                                          max_memory=max_memory)
        self.model.eval()
        off_gpu = sorted({str(p.device) for p in self.model.parameters() if p.device.type != "cuda"})
        self.offloaded = off_gpu
        if off_gpu and allow_offload:
            log(f"{model_id}: some weights are offloaded to {off_gpu} (allow_offload): a smoke test, slow")
        elif off_gpu:
            raise RuntimeError(f"{model_id} did not fit on the GPUs (some weights are on {off_gpu}); "
                               f"offloaded generation would be far too slow for a baseline run")
        self.end_ids = sorted({single_token_id(self.tok, END_OF_TURN_TOKENS, "end-of-turn"), self.tok.eos_token_id})
        self.input_device = self.model.get_input_embeddings().weight.device

    def chat_text(self, user_text: str, thinking: bool) -> str:
        """The exact string the model is given: its chat template applied to one user turn."""
        return self.tok.apply_chat_template([{"role": "user", "content": user_text}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=thinking)

    def generate(self, user_texts: list[str], thinking: bool, max_new_tokens: int,
                 decoding: Decoding = GREEDY) -> list[Reply]:
        """One reply per user text, generated together as one batch. `decoding`: see the module docstring."""
        import torch

        t0 = time.time()
        chats = [self.chat_text(t, thinking) for t in user_texts]
        # The chat template already contains the special tokens (e.g. <bos>); don't add them twice.
        batch = self.tok(chats, return_tensors="pt", padding=True, add_special_tokens=False).to(self.input_device)
        if decoding.mode == "sample":
            torch.manual_seed(decoding.batch_seed(user_texts))      # seeds the CPU and every GPU generator
            how = {"do_sample": True, "temperature": decoding.temperature, "top_p": decoding.top_p,
                   "top_k": decoding.top_k}
        else:
            how = {"do_sample": False}
        with torch.no_grad():
            out = self.model.generate(**batch, max_new_tokens=max_new_tokens, **how,
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
                                 hit_token_cap=end is None, seconds=seconds, generated_ids=ids))
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
            "offloaded_to": self.offloaded,      # non-empty only for a smoke test run with allow_offload
            "max_memory": self.max_memory,       # GPU weight budget set for allow_offload (None: device_map default)
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
