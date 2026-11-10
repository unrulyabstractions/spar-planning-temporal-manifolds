"""Model-layout checks for capture on weightless (meta-device) models built from cached configs; no GPU, no weights.

Covers what the hooks, the label-logit readout and the explicit device map rely on: where the decoder stack lives
(Qwen: `model.model`; Gemma 4 multimodal wrapper: `model.model.language_model`), the final logit soft cap, and that
the device map names every module.
"""
import os
import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402

from ptm.capture import ResidualHooks, explicit_device_map, final_logit_softcap, softcap, text_stack  # noqa: E402

MODELS = ["Qwen/Qwen3-14B", "Qwen/Qwen3-8B", "Qwen/Qwen3.5-9B", "Qwen/Qwen3.8-27B", "google/gemma-4-31B-it"]


@pytest.fixture(scope="module", params=MODELS)
def skeleton(request):
    try:
        cfg = AutoConfig.from_pretrained(request.param)
    except Exception as e:  # config not cached
        pytest.skip(f"{request.param} config not available offline: {e}")
    with torch.device("meta"):
        m = AutoModelForCausalLM.from_config(cfg)
    return request.param, cfg, m


def test_text_stack_and_hooks(skeleton):
    name, cfg, m = skeleton
    n = cfg.get_text_config().num_hidden_layers
    stack = text_stack(m)
    print(f"{name}: {type(m).__name__} -> stack {type(stack).__name__}, {len(stack.layers)} layers")
    assert len(stack.layers) == n
    if "gemma-4" in name:
        assert stack is m.model.language_model
    else:
        assert stack is m.model
    hooks = ResidualHooks(m)
    try:
        assert hooks.n_layers == n and hooks.modules[0] is stack.embed_tokens and hooks.modules[-1] is stack.layers[-1]
    finally:
        hooks.remove()


def test_final_logit_softcap(skeleton):
    name, cfg, m = skeleton
    cap = final_logit_softcap(m)
    print(f"{name}: final_logit_softcapping = {cap}")
    assert cap == (30.0 if "gemma-4" in name else None)


def test_softcap_matches_model_formula():
    z = torch.tensor([-80.0, -3.0, 0.0, 3.0, 55.0, 60.0])
    ref = torch.tanh(z / 30.0) * 30.0                     # Gemma4ForConditionalGeneration.forward: /c, tanh, *c
    assert torch.allclose(softcap(z, 30.0), ref)
    assert all(abs(softcap(float(v), 30.0) - float(r)) < 1e-5 for v, r in zip(z, ref))   # scalar path == tensor path
    assert softcap(z, None) is z and softcap(2.5, None) == 2.5
    # the cap compresses large logits, so the pairwise difference (hence p_short) changes when it is omitted
    d_raw, d_cap = 60.0 - 55.0, softcap(60.0, 30.0) - softcap(55.0, 30.0)
    print(f"logits 60 vs 55: raw difference {d_raw:.3f}, capped difference {d_cap:.3f}")
    assert d_cap < 0.5 * d_raw


def test_explicit_device_map(skeleton):
    name, cfg, m = skeleton
    n = cfg.get_text_config().num_hidden_layers
    split = n // 2
    dm = explicit_device_map(name, split)
    if "gemma-4" not in name:   # unchanged for Qwen: exactly the map used for every Qwen capture so far
        old = {"model.embed_tokens": 0, "model.rotary_emb": 0, "model.norm": 1, "lm_head": 1}
        old.update({f"model.layers.{i}": 0 if i < split else 1 for i in range(n)})
        assert dm == old
    names = [k for k, _ in m.named_parameters()] + [k for k, _ in m.named_buffers()]
    uncovered = [k for k in names if not any(k == p or k.startswith(p + ".") for p in dm)]
    print(f"{name}: device map has {len(dm)} entries; {len(names)} tensors, uncovered {uncovered[:5]}")
    assert not uncovered
    p = "model.language_model" if "gemma-4" in name else "model"
    assert dm[f"{p}.layers.{split - 1}"] == 0 and dm[f"{p}.layers.{split}"] == 1 and dm["lm_head"] == 1
