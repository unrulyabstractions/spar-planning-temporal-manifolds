"""The layer convention: a hook at layer L reads exactly hidden_states[L], and an edit at L is what the next
reader sees. Checked on a tiny random Qwen3 (always), on real Qwen3.5-0.8B weights (if cached and a GPU is
present: Qwen3.5 mixes linear- and full-attention blocks), and on the module tree of Gemma 4 31B (empty weights).
"""

import pytest
import torch

from steer.layers import capture, check_layer, decoder_layers, edit


def tiny_qwen3():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8)
    return Qwen3ForCausalLM(cfg).eval()


def assert_hooks_match_hidden_states(model, ids, layers):
    with torch.no_grad():
        ref = model(input_ids=ids, output_hidden_states=True).hidden_states
        with capture(model, layers) as acts:
            model(input_ids=ids)
    for layer in layers:
        diff = (acts[layer].float() - ref[layer].float()).abs().max().item()
        print(f"layer {layer}: max |hook - hidden_states[{layer}]| = {diff:.3g}")
        assert diff == 0


def test_hook_reads_hidden_states_index():
    model = tiny_qwen3()
    assert_hooks_match_hidden_states(model, torch.tensor([[1, 5, 9, 2, 7]]), [1, 2, 3])


def test_layers_outside_range_refused():
    model = tiny_qwen3()
    for bad in (0, 4, -1):
        with pytest.raises(ValueError):
            check_layer(model, bad)


def test_edit_is_seen_downstream_and_by_a_reader_at_the_same_layer():
    model = tiny_qwen3()
    ids = torch.tensor([[1, 5, 9, 2, 7]])
    v = torch.randn(32)
    with torch.no_grad():
        with capture(model, [2]) as before:
            base = model(input_ids=ids).logits
        with edit(model, 2, lambda h: h + v), capture(model, [2]) as after:
            edited = model(input_ids=ids).logits
        with edit(model, 2, lambda h: h):
            same = model(input_ids=ids).logits
    shift = (after[2] - before[2] - v).abs().max().item()
    print(f"reader at the edited layer sees +v: max error {shift:.3g}; logits moved by "
          f"{(edited - base).abs().max().item():.3g}; identity edit moved them by {(same - base).abs().max().item():.3g}")
    assert shift < 1e-6
    assert (edited - base).abs().max() > 1e-3
    assert torch.equal(same, base)


def test_gemma4_decoder_layers_found():
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM
    try:
        cfg = AutoConfig.from_pretrained("google/gemma-4-31B-it", local_files_only=True)
    except OSError:
        pytest.skip("Gemma 4 31B config not cached")
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(cfg)
    layers = decoder_layers(model)
    name = next(n for n, m in model.named_modules() if m is layers)
    print(f"Gemma 4 31B decoder blocks: {name} ({len(layers)}); layer 38 hooks block index 37")
    assert name == "model.language_model.layers" and len(layers) == 60


def test_real_qwen35_hybrid_blocks():
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    from transformers import AutoModelForCausalLM
    try:
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B", dtype=torch.bfloat16,
                                                     device_map="cuda:0", local_files_only=True).eval()
    except OSError:
        pytest.skip("Qwen3.5-0.8B not cached")
    ids = torch.tensor([[151644, 872, 198, 9707, 11, 1879, 0]], device="cuda:0")
    assert_hooks_match_hidden_states(model, ids, [1, 3, 4, 12, 22])
