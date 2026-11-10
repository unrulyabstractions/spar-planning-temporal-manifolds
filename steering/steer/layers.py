"""Which module is "layer L", and reading or editing the residual stream there.

Convention (the same as `output_hidden_states` and as Jared's code): **layer L is the residual stream after
decoder block L, counting blocks from 1.** So:

* layer L equals `hidden_states[L]` of a forward pass with `output_hidden_states=True`, for 1 <= L < n_layers;
* reading or editing layer L hooks the output of `decoder_layers(model)[L - 1]`;
* layer 0 (the embeddings) and layer n_layers are refused. In most Hugging Face models `hidden_states[n_layers]`
  is taken after the final norm, so a hook on the last block would not match it, and neither layer is a
  candidate for steering.

The report's "layer 38 of 60" on Gemma 4 31B is read in this convention: the output of the 38th block.
tests/test_layers.py checks that a hook at layer L returns exactly `hidden_states[L]`.

The decoder blocks are found without knowing the model class: they are the one `nn.ModuleList` whose length is
the text model's `num_hidden_layers` (`model.layers` in Qwen3 / Qwen3.5, `model.language_model.layers` in
Gemma 4). A model with zero or several such lists is refused rather than guessed.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator


def n_layers(model) -> int:
    """Number of decoder blocks of the text model."""
    config = model.config
    return getattr(config, "text_config", config).num_hidden_layers


def decoder_layers(model):
    """The model's decoder blocks, as an nn.ModuleList (see the module docstring)."""
    import torch

    n = n_layers(model)
    found = [(name, m) for name, m in model.named_modules() if isinstance(m, torch.nn.ModuleList) and len(m) == n]
    if len(found) != 1:
        raise ValueError(f"expected exactly one ModuleList of {n} decoder blocks, found {[name for name, _ in found]}")
    return found[0][1]


def check_layer(model, layer: int) -> None:
    """Refuse a layer outside 1 .. n_layers - 1 (see the module docstring)."""
    n = n_layers(model)
    if not 1 <= layer < n:
        raise ValueError(f"layer {layer} is outside 1..{n - 1} for this {n}-block model")


def _hidden(output):
    """The hidden states in a decoder block's output (some return a tensor, some a tuple led by it)."""
    return output[0] if isinstance(output, tuple) else output


@contextmanager
def capture(model, layers: list[int]) -> Iterator[dict]:
    """Record the residual stream at `layers` during the forward passes run inside the block.

        with capture(model, [38]) as acts:
            model(input_ids)
        acts[38]   # [batch, seq, d_model], detached, in the model's dtype

    Each forward pass overwrites the previous one's tensors.
    """
    blocks = decoder_layers(model)
    for layer in layers:
        check_layer(model, layer)
    store: dict = {}

    def recorder(layer: int):
        def hook(module, args, output):
            store[layer] = _hidden(output).detach()
        return hook

    handles = [blocks[layer - 1].register_forward_hook(recorder(layer)) for layer in layers]
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


@contextmanager
def edit(model, layer: int, fn: Callable) -> Iterator[None]:
    """Replace the residual stream at `layer` by `fn(hidden)` during the forward passes run inside the block.

    `fn` receives the [batch, seq, d_model] tensor and returns one of the same shape. Every position is edited,
    including those of the prompt (the report steers "all tokens"); restricting positions is up to `fn`.
    """
    check_layer(model, layer)

    def hook(module, args, output):
        if isinstance(output, tuple):
            return (fn(output[0]),) + tuple(output[1:])
        return fn(output)

    handle = decoder_layers(model)[layer - 1].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
