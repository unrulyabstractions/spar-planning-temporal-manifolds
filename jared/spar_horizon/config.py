"""Run settings shared by every experiment."""

from dataclasses import dataclass, asdict

import torch

from .full_bank import CHOICE_BANKS

BANKS = ("starter", "full", "units", "fill", *CHOICE_BANKS)


@dataclass
class Settings:
    model: str = "Qwen/Qwen3-8B"
    thinking: str = "on"            # "on" | "off"
    n_response: int = 6             # answer tokens kept after the suffix (or after </think>)
    max_think_tokens: int = 3072    # think budget; the block is force-closed past this
    batch_size: int = 12            # generation batch: 8B weights 16.4 GB + 12 x ~0.5 GB KV cache at 3.3k
                                    # tokens fits the 24 GB 4090; generate_batch steps down on OOM
    behavior_n: int | None = None   # order/label stability tests on an evenly spaced subset; None = all
    prefill: str = "I choose:"      # assistant prefill so the next token is the label; "" disables
    choice_prefix: str = ""         # with no prefill: the phrase the model writes before its label
                                    # ("I choose:" on the full bank); the label is located after it
    answer_window: int = 48         # tokens generated after the turn (or the think close) when the
                                    # label is located by choice_prefix rather than pinned by prefill
    bank: str = "starter"           # "starter" (the 116-prompt starter bank) | "full" (Alan's generator)
                                    # | "units" (full-bank prompts paired across units, 14_spec_unit_overlap.md)
                                    # | "fill" (crossed-unit prompts in the old banks' regime gaps, plus anchors,
                                    #   16_spec_pristine_manifold.md §4)
                                    # | "choice4" | "choice8" | "choice4_roi" | "choice8_roi" (k-option
                                    #   investment banks, 18_spec_multichoice.md)
    n_prompts: int = 500            # full bank size (the units bank has its own, UNITS_BASE pairs)
    limit: int | None = None        # prompt-bank subset for smoke tests
    seed: int = 0
    think_fractions: tuple = (0.25, 0.5, 1.0)  # kept positions inside the think block, as fractions
                                               # of each prompt's own think span; 0.75 dropped 2026-10-03
                                               # (weakest horizon signal in v3)
    cache_dir: str | None = None    # generation cache directory; None disables
    attn: str | None = None         # attention implementation passed to from_pretrained, e.g. "eager";
                                    # None keeps the library default (sdpa)

    def __post_init__(self):
        if self.thinking not in ("on", "off"):
            raise ValueError(f"thinking must be 'on' or 'off', got {self.thinking!r}")
        if self.bank not in BANKS:
            raise ValueError(f"bank must be one of {', '.join(map(repr, BANKS))}, got {self.bank!r}")
        if self.prefill and self.choice_prefix:
            raise ValueError("prefill and choice_prefix are exclusive: the label is either pinned or located")
        self.think_fractions = tuple(float(f) for f in self.think_fractions)
        if any(not 0 < f <= 1 for f in self.think_fractions):
            raise ValueError(f"think_fractions must lie in (0, 1], got {self.think_fractions}")
        if any(b <= a for a, b in zip(self.think_fractions, self.think_fractions[1:])):
            raise ValueError(f"think_fractions must be strictly increasing, got {self.think_fractions}")

    @property
    def dtype(self):
        gpu = torch.cuda.is_available() or (hasattr(torch, "xpu") and torch.xpu.is_available())
        return torch.bfloat16 if gpu else torch.float32

    @property
    def tag(self):
        """Run-folder suffix; the starter bank keeps the historical form."""
        bank = "" if self.bank == "starter" else f"_{self.bank}"
        return f"{self.model.split('/')[-1]}_think-{self.thinking}{bank}"

    def to_dict(self):
        return asdict(self)
