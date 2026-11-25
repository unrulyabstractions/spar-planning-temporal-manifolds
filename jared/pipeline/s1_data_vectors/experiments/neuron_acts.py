"""Spec 32 stage B: exact MLP neuron activations of Qwen3-8B at the suffix newline and
the pre-label token, on E0's own sequences (thinking on, full bank), for blocks 16, 22, 24.

`prepare` (local, tokenizer only) rebuilds each horizon prompt's sequence up to the
pre-label token from the generation cache, keeping only prompts whose cached trace is the
one E0 extracted (`steering_prepare.cached_reasoning`), and checks both positions against
E0's keep_idx. `run` (the box) does one forward pass per prompt and records, at both
positions and each block: the block input (hidden_states[L], to check against E0's saved
activations), the true MLP input (the residual entering post_attention_layernorm, to
measure stage A's approximation) and the neuron activations (the input of down_proj).
No generation.

    python -m pipeline.s1_data_vectors.experiments.neuron_acts prepare $RESULTS
    -> steering/neuron_lens_sequences.npz (gitignored; `just push` ships it)
    python -m pipeline.s1_data_vectors.experiments.neuron_acts run /workspace/results/horizon   # box, tmux, from jared/
    -> <root>/neuron_lens/neuron_acts_Qwen3-8B.npz
"""

import argparse
from pathlib import Path

import numpy as np

BLOCKS = (16, 22, 24)
POSITIONS = ("suffix newline", "pre-label")
MODEL = "Qwen/Qwen3-8B"
SEQUENCES = Path("steering/neuron_lens_sequences.npz")   # relative to jared/


def prepare(root):
    from transformers import AutoTokenizer

    from pipeline.s1_data_vectors.analysis.probe_dimensionality import positions
    from pipeline.s1_data_vectors.analysis.steering_prepare import cached_reasoning
    from spar_horizon.banks import build_records
    from spar_horizon.e0_runs import run_dir, run_settings

    root = Path(root)
    cfg = run_settings(root, MODEL, "on")
    records = build_records(cfg)
    z = np.load(run_dir(root, MODEL, "on") / "activations.npz")
    if not z["kept"].all():
        raise ValueError("E0 skipped prompts; rows of activations.npz no longer align with records")
    # rows of cell_data: the prompts with a horizon, in record order
    selected = np.flatnonzero(~np.isnan(z["horizons"]))
    meta = {"regions": {r: tuple(int(x) for x in z[f"region_{r}"]) for r in ("suffix", "think", "pre", "prelabel", "answer")},
            "tokens": [str(t) for t in z["tokens"]]}
    kept_index = [positions(meta)[p] for p in POSITIONS]
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    arrays, missing, mismatched = cached_reasoning(cfg, records, selected, tokenizer, z)
    rows = np.asarray(arrays["record"])            # index into `selected`, i.e. into cell_data rows
    sequence_ends = arrays["offsets"][1:] - arrays["offsets"][:-1]
    at = np.stack([z["keep_idx"][selected[r]][kept_index] for r in rows])
    if not np.array_equal(at[:, 1], sequence_ends - 1):
        raise ValueError("the pre-label token is not the last token before the label")
    out = SEQUENCES
    np.savez(out, ids=arrays["ids"], offsets=arrays["offsets"], positions=at, rows=rows)
    print(f"{len(rows)} of {len(selected)} horizon prompts; missing {len(missing)}, "
          f"mismatched {len(mismatched)}; median length {int(np.median(sequence_ends))} -> {out}")


def add_hooks(model, blocks):
    """Hooks that store, per block, its input, the MLP's input residual and the neuron
    activations (down_proj's input) of the last forward pass in the returned dict."""
    captured = {}

    def keep(name):
        def hook(module, args, kwargs):
            captured[name] = (args[0] if args else kwargs["hidden_states"])[0]
        return hook

    for b in blocks:
        layer = model.model.layers[b]
        layer.register_forward_pre_hook(keep(f"block_in {b}"), with_kwargs=True)
        layer.post_attention_layernorm.register_forward_pre_hook(keep(f"mlp_in {b}"), with_kwargs=True)
        layer.mlp.down_proj.register_forward_pre_hook(keep(f"acts {b}"), with_kwargs=True)
    return captured


def capture(model, captured, ids, at, blocks):
    """{name: [len(blocks), len(at), width] float32} at positions `at` of one sequence `ids` [T]."""
    import torch

    device = next(model.parameters()).device
    with torch.no_grad():
        model.model(input_ids=ids[None].to(device))   # the decoder only; the logits are not needed
    return {name: np.stack([captured[f"{name} {b}"][at.to(device)].float().cpu().numpy() for b in blocks])
            for name in ("block_in", "mlp_in", "acts")}


def run(root):
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM

    root = Path(root)
    seq = np.load(SEQUENCES)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
    captured = add_hooks(model, BLOCKS)
    per_prompt = []
    for i in tqdm(range(len(seq["rows"])), desc="forward", mininterval=10):
        ids = torch.from_numpy(seq["ids"][seq["offsets"][i]:seq["offsets"][i + 1]].astype(np.int64))
        per_prompt.append(capture(model, captured, ids, torch.from_numpy(seq["positions"][i].astype(np.int64)),
                                  BLOCKS))
    out = {name: np.stack([p[name] for p in per_prompt]) for name in per_prompt[0]}   # [n, block, position, width]
    path = root / "neuron_lens" / "neuron_acts_Qwen3-8B.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, rows=seq["rows"], blocks=np.array(BLOCKS), **out)
    print(f"wrote {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=("prepare", "run"))
    parser.add_argument("root", help="results root")
    args = parser.parse_args()
    prepare(args.root) if args.step == "prepare" else run(args.root)
