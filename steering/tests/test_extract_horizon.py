"""extract_horizon.py end to end on real Qwen3.5-0.8B (GPU; skipped without one or without the cached model)."""

import json

import numpy as np
import pytest
import torch

import extract_horizon
from fixtures import BANK
from pbank.records import load_prompts
from steer import activations
from steer.edits import LinearEdit
from steer.directions import random_directions
from steer.hooks import at_prompt, steering

N = 21      # the bank's first 21 prompts include two without a horizon (x0018-null, x0019-null)


@pytest.fixture(scope="module")
def lcm():
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    from pbank.local_model import LocalChatModel
    try:
        return LocalChatModel("Qwen/Qwen3.5-0.8B")
    except OSError:
        pytest.skip("Qwen3.5-0.8B not cached")


@pytest.fixture(scope="module")
def out(lcm, tmp_path_factory):
    root = tmp_path_factory.mktemp("acts")
    return extract_horizon.main(["--bank", str(BANK), "--model", "Qwen/Qwen3.5-0.8B", "--no-thinking", "--limit", str(N)],
                                activations_root=root, model=lcm)


def test_outputs_and_values(lcm, out):
    recs = load_prompts(BANK)[:N]
    meta = json.loads((out / "extract.json").read_text())
    sites = activations.horizon_sites(out)
    n_layers = lcm.model.config.get_text_config().num_hidden_layers
    print(f"{out.name}: counts {meta['counts']}; {len(meta['arrays'])} arrays for {n_layers} blocks; "
          f"first site {sites[recs[0].uid]['tokens']} at {sites[recs[0].uid]['token_indices']}")
    assert out.name == "Qwen3.5-0.8B-nothink__t0_s0__horizon"
    assert sorted(meta["arrays"]) == [f"L{layer:02d}_horizon.npy" for layer in range(n_layers + 1)]
    assert list(sites) == [r.uid for r in recs]
    nulls = [r.uid for r in recs if r.horizon is None]
    assert nulls and all(sites[u]["token_indices"] == [] and sites[u]["rows"] == [] for u in nulls)
    assert meta["counts"]["with_horizon"] == N - len(nulls) and meta["counts"]["decode_mismatches"] == 0
    assert meta["settings"]["thinking"] is False and meta["run_settings"]["model"] == "Qwen/Qwen3.5-0.8B"

    # Every stored row is the reference hidden state at that token, for a middle layer and for layer 0.
    for layer in (0, 12):
        acts = activations.load(out, layer, "horizon")          # also checks the array's recorded SHA-256
        worst = 0.0
        for r in (r for r in recs if r.horizon is not None):
            s = sites[r.uid]
            assert "".join(s["tokens"]).strip() == r.horizon and r.text[s["char_start"]:s["char_end"]] == r.horizon
            ids = lcm.tok(lcm.chat_text(r.text, False), add_special_tokens=False)["input_ids"]
            assert [ids[k] for k in s["token_indices"]] == s["token_ids"] and len(ids) == s["n_input_tokens"]
            with torch.no_grad():
                ref = lcm.model(input_ids=torch.tensor([ids], device=lcm.input_device),
                                output_hidden_states=True).hidden_states[layer][0, s["token_indices"]]
            worst = max(worst, float(np.abs(acts.X[s["rows"]] - ref.float().cpu().numpy()).max()))
            assert [acts.index[row]["token"] for row in s["rows"]] == s["tokens"]
        print(f"layer {layer}: {acts.X.shape}, max |stored - hidden_states[{layer}]| at the horizon tokens = {worst:.3g}")
        assert worst == 0


def test_refuses_an_existing_directory(lcm, out):
    with pytest.raises(SystemExit) as e:
        extract_horizon.main(["--bank", str(BANK), "--model", "Qwen/Qwen3.5-0.8B", "--no-thinking", "--out", str(out)],
                             model=lcm)
    assert "exists" in str(e.value.code)


def test_sites_drive_a_horizon_only_edit_in_batched_generation(lcm, out):
    # The per-prompt rows go to the hook as `rows`, their token_indices to at_prompt: in a left-padded batch, the
    # edited tokens must be exactly each prompt's horizon tokens, and a prompt without a horizon must be untouched.
    recs = load_prompts(BANK)[:N]
    sites = activations.horizon_sites(out)
    rows = [sites[r.uid] for r in recs]
    d = lcm.model.config.get_text_config().hidden_size
    record = []
    base = lcm.generate([r.text for r in recs], thinking=False, max_new_tokens=40)
    with steering(lcm.model, LinearEdit(random_directions(d, 1, 1)[0], 50.0), 4.0, 12,
                  at_prompt([s["token_indices"] for s in rows]), rows=rows, record=record):
        steered = lcm.generate([r.text for r in recs], thinking=False, max_new_tokens=40)
    prefill = record[0]["ctx"]
    batch = lcm.tok([lcm.chat_text(r.text, False) for r in recs], return_tensors="pt", padding=True,
                    add_special_tokens=False)["input_ids"]
    edited = [lcm.tok.decode(batch[i][prefill.mask[i].cpu()]) for i in range(N)]
    changed = sum(a.generated_ids != b.generated_ids for a, b in zip(base, steered))
    null_rows = [i for i, r in enumerate(recs) if r.horizon is None]
    print(f"edited tokens per prompt: {edited}; later passes edited {sum(int(r['ctx'].mask.sum()) for r in record[1:])}; "
          f"replies changed {changed}/{N}; rows passed to the edit: {prefill.rows[0]['uid']}..")
    assert [e.strip() for e in edited] == [r.horizon or "" for r in recs]
    assert all(not r["ctx"].mask.any() for r in record[1:]) and prefill.rows is rows
    assert len(null_rows) == 2 and changed > 0
    # A prompt without a horizon has no edited token, and rows of a batch do not interact, so its reply is identical.
    assert all(base[i].generated_ids == steered[i].generated_ids for i in null_rows)
