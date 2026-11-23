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
  3. effect:      a strong edit (x = --strong, at --positions) changes at least one reply, so the hook is applied;
                  the number of changed choices (parsed "I choose:") is reported beside the number of changed replies;
  4. method at 0: with --method, the bundled method at x = 0 (at --positions) must reproduce the unhooked ids, except
                  curve-snap, whose x = 0 is not the identity by design (it removes the off-curve offset): for it the
                  number of changed replies is reported, not checked.

--decoding (default greedy) sets how replies are generated; `sample` and `auto` are as in baseline.py, with
--temperature / --top-p / --top-k / --seed. A sampled batch is seeded from --seed and its own prompts, so the unhooked
and hooked runs draw the same random numbers and check 1 still demands identical token ids. Use the decoding of the
runs you are about to make.

--positions label / prelabel edit one token per row: the answer-label token of the row's unhooked reply (the site
extract.py reads as "label"), or the token before it ("prelabel", whose state predicts the label). The index comes
from the unhooked reply (steer.sites.locate_label), and two more checks apply to checks 3 and 4: exactly one token
per row is edited, and each steered reply equals the unhooked one up to and including the edited token (so the
edited token was the intended one). An edit at the label token cannot change the choice: the label is already
written when that state is computed.

Checks 1-2 use a random unit direction at strength 0, at all positions, so they test the hook itself. Without
--method, check 3 uses the same direction at --strong typical norms. With --method (a bundle from bundle.py), the
model and layer come from the bundle, and check 3 applies the method itself at x = --strong, in its own units
(strength for push methods, decades for curve slides). It uses the first --n prompts of the bank's extraction split,
writes nothing, and exits 1 if any check fails. Run it before the first steered generation on a new model, method or
transformers version.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pbank.answers import parse_choice  # noqa: E402
from pbank.local_model import LocalChatModel, add_decoding_args, decoding_from_args  # noqa: E402
from pbank.records import load_prompts  # noqa: E402
from steer.directions import random_directions  # noqa: E402
from steer.edits import ROLE_GENERATED, ROLE_PAD, ROLE_PROMPT, LinearEdit  # noqa: E402
from steer.hooks import at_generated, steering  # noqa: E402
from steer.sites import locate_label  # noqa: E402

SITE_OFFSET = {"label": 0, "prelabel": -1}      # generated index relative to the label token


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
    add_decoding_args(p, default="greedy")
    p.add_argument("--positions", default="all", choices=["all", "prompt", "generated", *SITE_OFFSET],
                   help="where checks 3 and 4 apply the edit (default all, as the bundles do)")
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

    records = [r for r in load_prompts(args.bank) if r.split == "extraction"][:args.n]
    texts = [r.text for r in records]
    t0 = time.time()
    lcm = LocalChatModel(args.model, allow_offload=args.allow_offload)
    d = lcm.model.config.get_text_config().hidden_size
    edit = LinearEdit(random_directions(d, 1, seed=0)[0], typical_norm=1.0)
    print(f"{args.model}: d = {d}, layer {args.layer}, {len(texts)} prompts, thinking {args.thinking}, "
          f"loaded in {time.time() - t0:.0f} s", flush=True)

    decoding = decoding_from_args(args, args.thinking, args.model)
    print(f"decoding {decoding.settings()}", flush=True)
    gen = lambda: lcm.generate(texts, thinking=args.thinking, max_new_tokens=args.max_new_tokens, decoding=decoding)
    base = gen()
    sites = None
    if args.positions in SITE_OFFSET:
        found = [locate_label(lcm.tok, r.generated_ids, tuple(rec.labels), args.thinking) for r, rec in zip(base, records)]
        if any(s is None for s in found):
            sys.exit(f"--positions {args.positions}: {sum(s is None for s in found)} unhooked replies have no label")
        sites = [s.label_index + SITE_OFFSET[args.positions] for s in found]
        print(f"{args.positions} token at generated index {sites} (label at {[s.label_index for s in found]})", flush=True)
    positions = at_generated(sites) if sites is not None else args.positions
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
    rec_strong: list = []
    with steering(lcm.model, strong_edit, args.strong, args.layer, positions, record=rec_strong):
        strong = gen()
    if method:
        rec_zero: list = []
        with steering(lcm.model, method.edit, 0.0, args.layer, positions, record=rec_zero):
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

    def site_check(name: str, rec: list, replies: list) -> None:
        """--positions label/prelabel: one edited token per row, and the replies agree with the unhooked ones up to it."""
        if sites is None:
            return
        per_row = sum(r["ctx"].mask.sum(1).cpu() for r in rec).tolist()
        prefix_ok = [a.generated_ids[:k + 1] == b.generated_ids[:k + 1] for a, b, k in zip(base, replies, sites)]
        print(f"   {name}: edited tokens per row {per_row}; replies equal to unhooked up to the edited token "
              f"{sum(prefix_ok)}/{len(prefix_ok)}", flush=True)
        if per_row != [1] * len(sites) or not all(prefix_ok):
            failures.append(f"{name}: not exactly the {args.positions} token was edited")

    choice = lambda reply, rec: parse_choice(reply.answer, tuple(rec.labels)).label
    changed = sum(a.generated_ids != b.generated_ids for a, b in zip(base, strong))
    moved = sum(choice(a, r) != choice(b, r) for a, b, r in zip(base, strong, records))
    what = (f"{method.meta['method']} at x = {args.strong} ({method.meta['request']['name']})" if method else
            f"strength {args.strong} x typical norm {norm:.3g}")
    print(f"3. effect: {what} at positions {args.positions} changed {changed}/{len(base)} replies, "
          f"{moved}/{len(base)} choices; first reply {(base[0].answer or '')[:60]!r} -> {(strong[0].answer or '')[:60]!r}",
          flush=True)
    site_check("strong edit", rec_strong, strong)
    if changed == 0:
        failures.append("a strong edit changed nothing: is the hook applied?")

    if method:
        moved = sum(a.generated_ids != b.generated_ids for a, b in zip(base, method_zero))
        snap = method.meta["method"] == "curve-snap"
        print(f"4. {method.meta['method']} at x = 0: {moved}/{len(base)} replies changed"
              + (" (reported only: snap at 0 removes the off-curve offset)" if snap else ""), flush=True)
        site_check("method at x = 0", rec_zero, method_zero)
        if moved and not snap:
            failures.append(f"{method.meta['method']} at x = 0 changed the output")

    print(f"{'FAILED: ' + '; '.join(failures) if failures else 'all checks passed'} ({time.time() - t0:.0f} s)", flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
