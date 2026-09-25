#!/usr/bin/env python
"""Cross-check the hook-based capture against transformers' own hidden_states and logits.

Checks, on a small batch:
  1. acts[l] == hidden_states[l] for l in 0..n_layers-1 (input to decoder layer l)
  2. acts[n_layers] == output of the last decoder layer (raw, pre-norm)
  3. norm(acts[n_layers][pos]) -> lm_head == model(...).logits[pos]
  4. the captured token at each stored position decodes to the expected string
"""
import argparse, sys
import pandas as pd, torch
from ptm.capture import CaptureConfig, ResidualHooks, capture_batch, load_model
from ptm.chat import encode_user_turn
from ptm.prompts import from_frame

ap = argparse.ArgumentParser(); ap.add_argument("--model", default="Qwen/Qwen3-8B"); ap.add_argument("--n", type=int, default=4)
a = ap.parse_args()
cfg = CaptureConfig(model_name=a.model, batch_size=a.n, max_new_tokens=40, n_response=8)
model, tok = load_model(cfg)
hooks = ResidualHooks(model)
samples = from_frame(pd.read_parquet("data/prompts/investment_n16_s1.parquet"))[: a.n]
with torch.no_grad():
    res = capture_batch(model, tok, hooks, samples, cfg)
hooks.remove()
L = hooks.n_layers
device = model.get_input_embeddings().weight.device
ok = True
torch.set_grad_enabled(False)
for s, r in zip(samples, res):
    enc = encode_user_turn(tok, s.text, cfg.enable_thinking)
    full = torch.tensor([enc.input_ids + r.gen_ids], device=device)
    # reference: library hidden states + full logits, single unpadded sequence
    ref = model(input_ids=full, output_hidden_states=True)
    hs = ref.hidden_states
    last_raw = {}
    h = model.model.layers[-1].register_forward_hook(lambda m, i, o: last_raw.setdefault("x", (o[0] if isinstance(o, (tuple, list)) else o)))
    model.model(input_ids=full); h.remove()
    n_trans = len(enc.transition_tokens)
    positions = list(range(enc.transition_start, enc.prompt_len)) + [enc.prompt_len + k for k in range(cfg.n_response) if k < len(r.gen_ids)]
    def rel(a, b):  # relative L2 error, per position, worst case
        return float(((a - b).norm(dim=-1) / b.norm(dim=-1)).max())
    def cos(a, b):
        return float(torch.nn.functional.cosine_similarity(a, b, dim=-1).min())
    worst_rel, worst_cos, shifted_cos = 0.0, 1.0, 1.0
    for l in range(L):
        refv = hs[l][0, positions].float().cpu()
        mine = r.acts[l, : len(positions)].float()
        worst_rel = max(worst_rel, rel(mine, refv)); worst_cos = min(worst_cos, cos(mine, refv))
        if l > 0:  # control: the same layer one position later must NOT match
            shifted_cos = min(shifted_cos, cos(mine, hs[l][0, [p + 1 for p in positions]].float().cpu()))
    ref_last = last_raw["x"][0, positions].float().cpu(); mine_last = r.acts[L, : len(positions)].float()
    rel_last, cos_last = rel(mine_last, ref_last), cos(mine_last, ref_last)
    normed_vs_raw = rel(hs[L][0, positions].float().cpu(), ref_last)
    line = (f"{s.sample_uid}: gen={r.gen_text[:48]!r} choice={r.choice} | layers0..{L-1}: worst rel err {worst_rel:.2e}, min cos {worst_cos:.5f} "
            f"(shifted-pos control min cos {shifted_cos:.3f}) | last: rel {rel_last:.2e} cos {cos_last:.5f} | hs[L] vs raw rel {normed_vs_raw:.2f}")
    if r.choice is not None:
        pos = enc.prompt_len + r.choice_gen_index - 1
        lg = ref.logits[0, pos].float().cpu()
        ia, ib = tok(" a)", add_special_tokens=False)["input_ids"][0], tok(" b)", add_special_tokens=False)["input_ids"][0]
        pr = torch.softmax(lg, -1)
        line += f" | logits ref a={lg[ia]:.3f} b={lg[ib]:.3f} mine a={r.logit_a:.3f} b={r.logit_b:.3f}; p ref a={pr[ia]:.3f} b={pr[ib]:.3f}"
        ok &= abs((lg[ia] - lg[ib]) - (r.logit_a - r.logit_b)) < 1.0   # bf16 ulp at |logit|~60 is 0.5
        # the token at the choice position must be the label's first token
        assert full[0, pos + 1].item() in (ia, ib)
    toks_at = [tok.decode([full[0, p].item()]) for p in positions]
    assert toks_at[:n_trans] == enc.transition_tokens == r.transition_tokens, toks_at
    assert toks_at[n_trans:] == r.response_tokens[: len(toks_at) - n_trans], (toks_at[n_trans:], r.response_tokens)
    ok &= worst_rel < 2e-2 and worst_cos > 0.999 and rel_last < 2e-2 and shifted_cos < 0.98
    print(line); sys.stdout.flush()
print("mem per gpu (GiB):", [round(torch.cuda.max_memory_allocated(i) / 2**30, 2) for i in range(torch.cuda.device_count())])
print("VERIFY", "OK" if ok else "FAILED")
