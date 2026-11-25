import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import Qwen3Config, Qwen3ForCausalLM

from pipeline.s1_data_vectors.experiments.neuron_acts import add_hooks, capture
from pipeline.s1_data_vectors.analysis.neuron_lens_compare import (centre_within, neuron_activations,
                                                                   spearman_columns)


def test_neuron_activations_match_the_mlp_on_its_true_input():
    """Given the residual entering post_attention_layernorm, the formula reproduces the
    input of down_proj (the neuron activations) of a real Qwen3 block."""
    torch.manual_seed(0)
    config = Qwen3Config(hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=4,
                         num_key_value_heads=2, vocab_size=50, head_dim=8, rms_norm_eps=1e-6)
    model = Qwen3ForCausalLM(config).eval()
    block = model.model.layers[1]
    with torch.no_grad():   # non-trivial norm weights, so the test checks they are applied
        block.post_attention_layernorm.weight.uniform_(0.5, 1.5)
    seen = {}
    norm_hook = block.post_attention_layernorm.register_forward_hook(
        lambda m, args, out: seen.__setitem__("h", args[0].detach()))
    down_hook = block.mlp.down_proj.register_forward_pre_hook(
        lambda m, args: seen.__setitem__("acts", args[0].detach()))
    with torch.no_grad():
        model(input_ids=torch.randint(0, 50, (1, 7)))
    norm_hook.remove()
    down_hook.remove()

    weights = [p.detach().numpy() for p in (block.mlp.gate_proj.weight, block.mlp.up_proj.weight,
                                            block.post_attention_layernorm.weight)]
    ours = neuron_activations(seen["h"][0].numpy(), *weights, eps=config.rms_norm_eps)
    np.testing.assert_allclose(ours, seen["acts"][0].numpy(), rtol=1e-4, atol=1e-6)


def test_stage_b_capture_matches_hidden_states_and_the_formula():
    """The box hooks: block_in equals hidden_states[b], and acts equal the A1 formula on mlp_in."""
    torch.manual_seed(0)
    config = Qwen3Config(hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
                         num_key_value_heads=2, vocab_size=50, head_dim=8, rms_norm_eps=1e-6)
    model = Qwen3ForCausalLM(config).eval()
    blocks, ids, at = (1, 3), torch.randint(0, 50, (9,)), torch.tensor([2, 8])
    out = capture(model, add_hooks(model, blocks), ids, at, blocks)
    with torch.no_grad():
        hidden = model(input_ids=ids[None], output_hidden_states=True).hidden_states
    for j, b in enumerate(blocks):
        np.testing.assert_allclose(out["block_in"][j], hidden[b][0, at].numpy(), rtol=1e-5, atol=1e-6)
        mlp = model.model.layers[b].mlp
        weights = [p.detach().numpy() for p in (mlp.gate_proj.weight, mlp.up_proj.weight,
                                                model.model.layers[b].post_attention_layernorm.weight)]
        np.testing.assert_allclose(out["acts"][j], neuron_activations(out["mlp_in"][j], *weights),
                                   rtol=1e-4, atol=1e-6)
    assert not np.allclose(out["mlp_in"], out["block_in"])   # attention was added in between


def test_spearman_columns_matches_scipy():
    rng = np.random.default_rng(0)
    A, y = rng.standard_normal((40, 5)), rng.standard_normal(40)
    A[:, 2] = y ** 3   # a monotone column: rho exactly 1
    expected = [spearmanr(A[:, j], y)[0] for j in range(5)]
    np.testing.assert_allclose(spearman_columns(A, y), expected, atol=1e-12)


def test_centre_within_removes_each_group_mean():
    values = np.array([[1.0], [3.0], [10.0], [14.0]])
    groups = np.array(["a", "a", "b", "b"])
    np.testing.assert_allclose(centre_within(values, groups)[:, 0], [-1, 1, -2, 2])
