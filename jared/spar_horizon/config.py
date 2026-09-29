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

    def __post_init__(self):
        if self.thinking not in ("on", "off"):
            raise ValueError(f"thinking must be 'on' or 'off', got {self.thinking!r}")

    @property
    def dtype(self):
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32

    @property
    def tag(self):
        return f"{self.model.split('/')[-1]}_think-{self.thinking}"

    def to_dict(self):
        return asdict(self)
