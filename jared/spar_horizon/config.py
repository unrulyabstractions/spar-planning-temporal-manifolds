"""Run settings shared by every experiment."""

from dataclasses import dataclass, asdict

import torch


@dataclass
class Settings:
    model: str = "Qwen/Qwen3-8B"
    thinking: str = "on"            # "on" | "off"
    n_response: int = 6             # answer tokens kept after the suffix (or after </think>)
    max_think_tokens: int = 3072    # think budget; the block is force-closed past this
    batch_size: int = 8             # generation batch; 8 keeps the 8B + 3k-token KV cache under 24 GB
    prefill: str = "I choose:"      # assistant prefill so the next token is the label; "" disables
    limit: int | None = None        # prompt-bank subset for smoke tests
    seed: int = 0
    think_fractions: tuple = (0.25, 0.5, 0.75, 1.0)  # kept positions inside the think block, as
                                                     # fractions of each prompt's own think span
    cache_dir: str | None = None    # generation cache directory; None disables
    attn: str | None = None         # attention implementation passed to from_pretrained, e.g. "eager";
                                    # None keeps the library default (sdpa)

    def __post_init__(self):
        if self.thinking not in ("on", "off"):
            raise ValueError(f"thinking must be 'on' or 'off', got {self.thinking!r}")
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
        return f"{self.model.split('/')[-1]}_think-{self.thinking}"

    def to_dict(self):
        return asdict(self)
