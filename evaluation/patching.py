"""Activation patching at the horizon: how far an edit at the horizon's tokens could move the answer.

The source of a patch is the same prompt with its horizon scaled by 10^(±Δ) in the same unit ("7 years" -> "70
years" at +1 decade, "0.7 years" at -1). Its states at layer L are put into the target prompt's replay, where the
target's own reasoning is kept (teacher forcing), so the readout is replay_eval's y = logit(far) - logit(near):

  text     no hook: the target's prompt with the source's horizon text, and the target's reasoning. The most a change
           of the stated horizon can do while the reasoning is held fixed.
  full     h <- h_src at the patched positions (every dimension): the ceiling for any edit of those tokens at layer L.
  subspace h <- h + P (h_src - h), P the orthogonal projection onto a basis (a fitted direction, a curve's PCs, or
           random directions for the null): how much of `full` that subspace carries.

Like the steering arms, each runs at +x (source 10^Δ times longer) and -x (10^Δ times shorter), so analyze.py's odd /
even / meaning / letter split and the switch-point shift apply unchanged. The reasoning stays the target's, so even a
perfect horizon edit does not move the switch point by Δ decades: patch:text measures what a real horizon change does
with the reasoning held fixed (2026-11-28, Qwen3-8B: meaning +0.52 logits, switch point unchanged).

Which positions (the replay's write span):
  horizon       the last token of the horizon phrase (the unit, " years"). Only that token can be aligned: numbers are
                one token per digit, so "7 years" and "70 years" have different lengths. It is also the first token
                whose state has read the whole phrase.
  from_horizon  that token through the end of the prompt. The text after the horizon is the same in source and target,
                so the positions align from the end. Reasoning tokens are never patched: the source has no reasoning.
A source state depends only on the tokens before it (causal attention), so the source needs one forward pass over its
prompt, not a generation.
"""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import torch

from steer.sites import horizon_char_span, locate_horizon

PATCH_SPANS = {"horizon": "horizon_last", "from_horizon": "from_horizon_last"}   # replay write span -> patched tokens


# ------------------------------------------------------------------ the source prompt

def scaled_horizon(horizon: str, decades: int) -> str:
    """The horizon with its number times 10^decades, in the same unit: ("7 years", 1) -> "70 years",
    ("10.32 years", -1) -> "1.032 years". Exact decimal arithmetic, so no float rounding enters the text."""
    number, unit = horizon.split(" ", 1)
    value = Decimal(number).scaleb(decades)
    return f"{value.normalize():f} {unit}"


def with_horizon(text: str, horizon: str, new: str) -> str:
    """The prompt with the CONSTRAINT line's horizon replaced (an option can state the same duration; it is kept)."""
    start, end = horizon_char_span(text, horizon)
    return text[:start] + new + text[end:]


def span_indices(span: str, site, prompt_len: int) -> list[int]:
    """Token indices (into the model input) of a write span for one prompt with horizon site `site`
    (steer.sites.HorizonSite). Steering spans: horizon (the phrase), from_horizon (the phrase through the prompt's
    last token). Patch spans (PATCH_SPANS): the phrase's last token, or it through the prompt's last token."""
    first, last = site.token_indices[0], site.token_indices[-1]
    return {"horizon": list(site.token_indices),
            "from_horizon": list(range(first, prompt_len)),
            "horizon_last": [last],
            "from_horizon_last": list(range(last, prompt_len))}[span]


def find_horizon(tok, chat: str, text: str, horizon: str):
    """locate_horizon, with extract_horizon.py's audit: the tokens found must spell the horizon exactly."""
    site = locate_horizon(tok, chat, text, horizon)
    if site.decoded.strip() != horizon:
        raise ValueError(f"the horizon's tokens decode to {site.decoded!r}, not {horizon!r}")
    return site


def token_mask(indices: list[list[int]]):
    """A steering `positions` callable for one right-padded teacher-forced pass: row i's tokens indices[i]. With right
    padding and one pass from position 0, a token's position is its index in the row's sequence."""

    def mask(ctx) -> torch.Tensor:
        if ctx.position.shape[0] != len(indices):
            raise ValueError(f"token_mask: {len(indices)} index lists for a batch of {ctx.position.shape[0]}")
        out = torch.zeros_like(ctx.position, dtype=torch.bool)
        for i, row in enumerate(indices):
            out[i, row] = True
        return out

    return mask


# ------------------------------------------------------------------ bases

def orthonormal_rows(V: np.ndarray) -> np.ndarray:
    """[k, d] orthonormal rows spanning the rows of V (k <= d)."""
    V = np.atleast_2d(np.asarray(V, dtype=np.float64))
    return np.linalg.qr(V.T)[0].T


def method_basis(method) -> np.ndarray:
    """The subspace a fitted method acts in: a linear method's direction (k = 1), a curve method's PCs (k components)."""
    if method.kind == "push":
        return orthonormal_rows(method.edit.direction)
    U = np.load(method.path / "arrays.npz")["U"]
    return orthonormal_rows(U if U.shape[0] < U.shape[1] else U.T)


def draw_basis(draws: np.ndarray, k: int, i: int) -> np.ndarray:
    """Null basis i of dimension k from a family of shared random unit vectors [n, d]: rows i*k .. i*k+k-1,
    orthonormalized (k = 1: draw i itself)."""
    if (i + 1) * k > len(draws):
        raise ValueError(f"basis {i} of dimension {k} needs {(i + 1) * k} draws; the family has {len(draws)}")
    return orthonormal_rows(draws[i * k:(i + 1) * k])


# ------------------------------------------------------------------ the edits

class TextEdit:
    """Not a hook: the replay rewrites each prompt's horizon to 10^(±|x|) times itself (x in decades, a whole number)
    and keeps the target's reasoning."""

    kind = "text"

    def prompt_text(self, text: str, horizon: str, x: float) -> str:
        decades = int(round(x))
        if decades != x:
            raise ValueError(f"the text arm scales by whole decades, not {x}")
        return with_horizon(text, horizon, scaled_horizon(horizon, decades))


class PatchEdit:
    """h <- h + |x| * P (h_src - h) at each row's patched positions, h_src from the source prompt 10^sign(x) times
    longer (decades = 1 for |x| = 1). basis None: P = identity (the full patch). x = 0 is an exact no-op.

    The replay calls prepare(rep, items) before its batched passes, outside any hook, to run the source prompts;
    then delta() looks each row up by uid in ctx.rows, and its positions are ctx.mask's (the patch span)."""

    kind = "patch"

    def __init__(self, basis: np.ndarray | None = None, decades: int = 1):
        self.basis, self.decades = basis, decades
        self.sources: dict = {}            # (uid, sign) -> float32 [positions, d] on the model's device
        self.where = None                  # (span, layer) the sources were read at
        self._U = None

    def prepare(self, rep, items, span: str, layer: int) -> None:
        if self.where not in (None, (span, layer)):
            raise ValueError(f"this patch's sources are at {self.where}, not ({span!r}, {layer}): make a new PatchEdit")
        self.where = (span, layer)
        todo = [it for it in items if (it["uid"], 1) not in self.sources]
        if not todo:
            return
        for sign in (1, -1):
            states = rep.source_states(todo, self.decades * sign, span, layer)
            for it, s in zip(todo, states):
                self.sources[(it["uid"], sign)] = s

    def delta(self, h: torch.Tensor, x, ctx) -> torch.Tensor:
        out = torch.zeros(h.shape, dtype=torch.float32, device=h.device)
        if float(x) == 0:
            return out
        if abs(float(x)) != 1:
            raise ValueError(f"a patch runs at x = ±1 (source 10^±{self.decades} times the horizon), not {x}")
        sign = 1 if float(x) > 0 else -1
        if self.basis is not None and self._U is None:
            self._U = torch.as_tensor(self.basis, dtype=torch.float32, device=h.device)
        for i, row in enumerate(ctx.rows):
            pos = ctx.mask[i].nonzero().squeeze(1)
            src = self.sources[(row["uid"], sign)]
            if len(pos) != len(src):
                raise ValueError(f"{row['uid']}: {len(pos)} patched positions but {len(src)} source states")
            diff = src - h[i, pos].float()
            out[i, pos] = diff if self._U is None else (diff @ self._U.T) @ self._U
        return out
