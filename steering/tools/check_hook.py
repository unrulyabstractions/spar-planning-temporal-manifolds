"""Gate before steering a model: does the hook behave on THIS model, in the exact generation setup?

    python tools/check_hook.py --model google/gemma-4-31B-it --layer 38 --bank ../prompt_bank/data/t0_s0 --n 8
    python tools/check_hook.py --model Qwen/Qwen3.5-0.8B --layer 12 --bank ../prompt_bank/data/t0_s0 --no-thinking
    python tools/check_hook.py --method methods/gemma-4-31B-it/caa__... --bank ../prompt_bank/data/t0_s0

tests/test_hooks.py checks the hook on a tiny Qwen3 and on Qwen3.5-0.8B. This script repeats the checks that depend on
the model class and its generate() (which arguments reach the forward pass, how padding is laid out), on any model,
through pbank's LocalChatModel with batched greedy generation:

  1. zero edit:   a hooked run with strength 0 gives exactly the unhooked token ids, for every prompt (null N0);
  2. roles:       the first pass is the prompt (left padding where the attention mask is 0), every later pass is one
                  generated token per row, and positions advance by one per pass;
  3. effect:      a strong edit (x = --strong, all positions) changes at least one reply, so the hook is applied;
  4. method at 0: with --method, the bundled method at x = 0 must reproduce the unhooked ids exactly, except
                  curve-snap, whose x = 0 is not the identity by design (it removes the off-curve offset): for it the
                  number of changed replies is reported, not checked.

Checks 1-2 use a random unit direction at strength 0, so they test the hook itself. Without --method, check 3 uses
the same direction at --strong typical norms. With --method (a bundle from bundle.py), the model and layer come from
the bundle, and check 3 applies the method itself at x = --strong, in its own units (strength for push methods,
decades for curve slides). It uses the first --n prompts of the bank's extraction split, writes nothing, and exits 1 if
any check fails. Run it before the first steered generation on a new model, method or transformers version.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pbank.local_model import LocalChatModel  # noqa: E402
from pbank.records import load_prompts  # noqa: E402
from steer.directions import random_directions  # noqa: E402
from steer.edits import ROLE_GENERATED, ROLE_PAD, ROLE_PROMPT, LinearEdit  # noqa: E402
from steer.hooks import steering  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help="required without --method")
    p.add_argument("--layer", type=int, help="required without --method")
    p.add_argument("--method", type=Path, help="a method bundle (bundle.py): checks it, at its model and layer")
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--n", type=int, default=8, help="prompts, generated as one batch")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=None,
                   help="default: as the method was fitted with --method, else on")
    p.add_argument("--strong", type=float, default=4.0, help="strength of the effect check, in typical norms")
    p.add_argument("--allow-offload", action="store_true",
                   help="let weights that don't fit on the GPUs stay in CPU RAM (a local smoke test of a big model)")
    args = p.parse_args()
    method = None
    if args.method:
        from steer.methods import load
        method = load(args.method)
        model_id, layer = method.meta["fitted_on"]["model"]["model_id"], method.layer
        if (args.model and args.model != model_id) or (args.layer and args.layer != layer):
            sys.exit(f"{args.method} was fitted on {model_id} layer {layer}; drop --model/--layer or match them")
        args.model, args.layer = model_id, layer
        if args.thinking is None:
            args.thinking = bool(method.meta["fitted_on"]["thinking"])
    elif not (args.model and args.layer):
        sys.exit("give --model and --layer, or --method")
    if args.thinking is None:
        args.thinking = True

    texts = [r.text for r in load_prompts(args.bank) if r.split == "extraction"][:args.n]
    t0 = time.time()
    lcm = LocalChatModel(args.model, allow_offload=args.allow_offload)
    d = lcm.model.config.get_text_config().hidden_size
    edit = LinearEdit(random_directions(d, 1, seed=0)[0], typical_norm=1.0)
    print(f"{args.model}: d = {d}, layer {args.layer}, {len(texts)} prompts, thinking {args.thinking}, "
          f"loaded in {time.time() - t0:.0f} s", flush=True)

    gen = lambda: lcm.generate(texts, thinking=args.thinking, max_new_tokens=args.max_new_tokens)
    base = gen()
    record: list = []
    with steering(lcm.model, edit, 0.0, args.layer, "all", record=record):
        zero = gen()
    # The strong check's unit: the mean activation norm at this layer over the first prompt's tokens.
    with torch.no_grad():
        from steer.layers import capture
        batch = lcm.tok([lcm.chat_text(texts[0], args.thinking)], return_tensors="pt",
                        add_special_tokens=False).to(lcm.input_device)
        with capture(lcm.model, [args.layer]) as acts:
            lcm.model(**batch)
        norm = acts[args.layer].float().norm(dim=-1).mean().item()
    strong_edit = method.edit if method else LinearEdit(edit.direction, typical_norm=norm)
    with steering(lcm.model, strong_edit, args.strong, args.layer, "all"):
        strong = gen()
    if method:
        with steering(lcm.model, method.edit, 0.0, args.layer, "all"):
            method_zero = gen()

    failures = []
    same = [a.generated_ids == b.generated_ids for a, b in zip(base, zero)]
    print(f"1. zero edit: identical token ids for {sum(same)}/{len(same)} prompts", flush=True)
    if not all(same):
        failures.append("zero edit changed the output")

    prefill, decode = record[0]["ctx"], [r["ctx"] for r in record[1:]]
    pads = (prefill.role == ROLE_PAD).sum(1).tolist()
    roles_ok = (bool(((prefill.role == ROLE_PROMPT) | (prefill.role == ROLE_PAD)).all())
                and all(c.role.shape[1] == 1 and bool((c.role == ROLE_GENERATED).all()) for c in decode)
                and [int(c.position[0, 0]) for c in decode] == list(range(prefill.role.shape[1], prefill.role.shape[1] + len(decode))))
    lead_pads_ok = all(bool((prefill.role[i, :n] == ROLE_PAD).all()) and bool((prefill.role[i, n:] == ROLE_PROMPT).all())
                       for i, n in enumerate(pads))
    print(f"2. roles: prefill {tuple(prefill.role.shape)} with left padding per row {pads}; {len(decode)} decode passes "
          f"of one generated token each: {'ok' if roles_ok and lead_pads_ok else 'WRONG'}", flush=True)
    if not (roles_ok and lead_pads_ok):
        failures.append("roles or positions are not as expected")

    changed = sum(a.generated_ids != b.generated_ids for a, b in zip(base, strong))
    what = (f"{method.meta['method']} at x = {args.strong} ({method.meta['request']['name']})" if method else
            f"strength {args.strong} x typical norm {norm:.3g}")
    print(f"3. effect: {what} changed {changed}/{len(base)} replies; first reply "
          f"{(base[0].answer or '')[:60]!r} -> {(strong[0].answer or '')[:60]!r}", flush=True)
    if changed == 0:
        failures.append("a strong edit changed nothing: is the hook applied?")

    if method:
        moved = sum(a.generated_ids != b.generated_ids for a, b in zip(base, method_zero))
        snap = method.meta["method"] == "curve-snap"
        print(f"4. {method.meta['method']} at x = 0: {moved}/{len(base)} replies changed"
              + (" (reported only: snap at 0 removes the off-curve offset)" if snap else ""), flush=True)
        if moved and not snap:
            failures.append(f"{method.meta['method']} at x = 0 changed the output")

    print(f"{'FAILED: ' + '; '.join(failures) if failures else 'all checks passed'} ({time.time() - t0:.0f} s)", flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
