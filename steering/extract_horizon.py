"""Read the residual stream at the tokens that state the horizon ("7 years"), at every layer, for every prompt.

    python extract_horizon.py --bank ../prompt_bank/data/t0_s0 --model google/gemma-4-31B-it
    python extract_horizon.py --bank ../prompt_bank/data/t0_s0 --model Qwen/Qwen3.5-2B --no-thinking --layers 12 15

Why a separate script from extract.py: the horizon is in the prompt, so its activations do not depend on the reply.
This needs no baseline run and generates nothing. It is one forward pass per prompt (batch 1, so no padding) over
the model input, which is the chat template applied to the prompt, exactly as baseline.py and extract.py build it.

What is kept, per prompt that states a horizon: the activation at each token of the horizon, separately, with its
token string. "15 minutes" is three tokens on Gemma 4 and Qwen3.5 ("1", "5", " minutes"): numbers are split into
digits, and the unit is its own token. The tokens are found by `steer.sites.locate_horizon`, from the horizon's one
occurrence in the CONSTRAINT line.

The audit is part of the run: for every prompt the kept tokens must decode back to the stated horizon, exactly
(unit rewrites such as "100 weeks" included). If any does not, the script lists them and writes nothing.

Layers (--layers, default all): 0 .. n_layers, in the convention of steer/layers.py. Layer L >= 1 is the residual
stream after decoder block L. Layer 0 is the input of block 1 (the embeddings). Layer n_layers is the output of the
last block before the final norm, which is not `hidden_states[n_layers]` (see `steer.layers.capture_all`).

Thinking (--thinking, the default, or --no-thinking) is part of the model input: Gemma 4's template puts a system
turn before the prompt when thinking is on, so the horizon tokens' indices and activations depend on it. Use the
setting of the runs these activations will be used with; it is recorded.

Which prompts: --split all (default), extraction or verification. No-horizon (null) prompts have no horizon tokens:
they get a row in horizon_sites.jsonl with empty lists, and no activations.

Output: activations/<model>[-nothink]__<bank>__horizon/ unless --out is given (format in steer/activations.py):

    extract.json          settings, model and software, the bank's identity, counts, array hashes
    index.jsonl           one row per array row = one horizon token of one prompt: uid, which token of the horizon,
                          its index in the model input, its id and string
    horizon_sites.jsonl   one row per prompt: where its horizon is (characters, token indices, token strings), its
                          array rows, and the per-prompt facts an edit may need (option order, label style, log-time)
    L<layer>_horizon.npy  float32 [rows, d], one file per layer

An existing output directory is refused: an extraction is a record, so redo it in a new directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from pbank.local_model import LocalChatModel, log
from pbank.provenance import utc_now
from pbank.records import load_prompts, prompts_sha256, write_jsonl
from steer.activations import HORIZON, array_name, file_sha256
from steer.fitting import log10_days
from steer.layers import capture_all, n_layers
from steer.provenance import code_commit
from steer.sites import locate_horizon

HERE = Path(__file__).resolve().parent


def default_name(model_id: str, thinking: bool, bank: str) -> str:
    return f"{model_id.split('/')[-1]}{'' if thinking else '-nothink'}__{bank}__horizon"


def main(argv: list[str] | None = None, activations_root: Path = HERE / "activations", model=None) -> Path:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bank", type=Path, required=True, help="bank directory (../prompt_bank/data/<bank>)")
    p.add_argument("--model", required=True, help="Hugging Face model id")
    p.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True,
                   help="build the model input with thinking on (default) or off; see above")
    p.add_argument("--layers", type=int, nargs="+", help="layers to keep (default: all, 0 .. n_layers)")
    p.add_argument("--split", default="all", choices=["extraction", "verification", "all"])
    p.add_argument("--out", type=Path, help="output directory (default: activations/<model>[-nothink]__<bank>__horizon)")
    p.add_argument("--limit", type=int, help="only the first N prompts (smoke tests)")
    p.add_argument("--allow-offload", action="store_true",
                   help="smoke tests only: let weights that don't fit on the GPUs stay in CPU RAM (recorded)")
    args = p.parse_args(argv)

    records = load_prompts(args.bank)
    out = args.out or Path(activations_root) / default_name(args.model, args.thinking, args.bank.name)
    if out.exists():
        sys.exit(f"{out} exists; extractions are records, so write a new one with --out")
    selected = [r for r in records if args.split == "all" or r.split == args.split][:args.limit]

    # `model` is for tests: an object with the parts of LocalChatModel used below.
    model = model or LocalChatModel(args.model, allow_offload=args.allow_offload)
    total = n_layers(model.model)
    layers = sorted(set(args.layers)) if args.layers else list(range(total + 1))
    if not all(0 <= layer <= total for layer in layers):
        sys.exit(f"--layers must be within 0..{total} for this {total}-block model")

    # Pass 1, no model: find every horizon's tokens and audit them before any GPU time is spent.
    sites, failures = {}, []
    for rec in selected:
        if rec.horizon is None:
            continue
        chat = model.chat_text(rec.text, args.thinking)
        site = locate_horizon(model.tok, chat, rec.text, rec.horizon)
        ids = model.tok(chat, add_special_tokens=False)["input_ids"]
        if site.decoded.strip() != rec.horizon or [ids[k] for k in site.token_indices] != site.token_ids:
            failures.append(f"{rec.uid}: horizon {rec.horizon!r}, tokens {site.tokens}")
        sites[rec.uid] = (site, ids)
    if failures:
        sys.exit(f"{len(failures)} of {len(sites)} horizons do not decode back from their tokens; nothing written:\n  "
                 + "\n  ".join(failures[:20]))
    if not sites:
        sys.exit("no selected prompt states a horizon")
    log(f"{args.model}, thinking {'on' if args.thinking else 'off'}: {len(selected)} prompts of {args.bank.name}, "
        f"{len(sites)} with a horizon, {sum(len(s.token_indices) for s, _ in sites.values())} horizon tokens, "
        f"all decode back to the stated horizon; layers {layers[0]}..{layers[-1]} ({len(layers)})")

    # Pass 2: one forward pass per prompt.
    import torch

    started = utc_now()
    index, site_rows = [], []
    arrays = {layer: [] for layer in layers}
    for n, rec in enumerate(selected, 1):
        facts = {"uid": rec.uid, "split": rec.split, "regime": rec.regime, "horizon": rec.horizon,
                 "horizon_grid_value": rec.horizon_grid_value,
                 "log10_days": None if rec.horizon is None else round(log10_days(rec.horizon_grid_value), 6),
                 "near_first": rec.near_first, "label_style": "/".join(rec.labels), "far_label": rec.far_label}
        if rec.uid not in sites:
            site_rows.append({**facts, "char_start": None, "char_end": None, "n_input_tokens": None,
                              "token_indices": [], "token_ids": [], "tokens": [], "rows": []})
            continue
        site, ids = sites[rec.uid]
        x = torch.tensor([ids], device=model.input_device)
        with torch.no_grad(), capture_all(model.model) as acts:
            try:
                model.model(input_ids=x, use_cache=False, logits_to_keep=1)
            except TypeError:   # a model class without logits_to_keep
                model.model(input_ids=x, use_cache=False)
        for layer in layers:
            arrays[layer].append(acts[layer][0, site.token_indices].float().cpu().numpy())
        first = len(index)
        for k, (pos, tid, token) in enumerate(zip(site.token_indices, site.token_ids, site.tokens)):
            index.append({"row": len(index), "uid": rec.uid, "k": k, "n_tokens": len(site.tokens),
                          "token_index": pos, "token_id": tid, "token": token})
        site_rows.append({**facts, "char_start": site.char_start, "char_end": site.char_end,
                          "n_input_tokens": site.n_input_tokens, "token_indices": site.token_indices,
                          "token_ids": site.token_ids, "tokens": site.tokens,
                          "rows": list(range(first, len(index)))})
        if n % 50 == 0 or n == len(selected):
            log(f"  [{n}/{len(selected)}] {len(index)} horizon tokens")

    out.mkdir(parents=True)
    write_jsonl(out / "index.jsonl", index)
    write_jsonl(out / "horizon_sites.jsonl", site_rows)
    hashes = {}
    for layer, chunks in arrays.items():
        name = array_name(layer, HORIZON)
        X = np.concatenate(chunks).astype(np.float32)
        np.save(out / name, X)
        hashes[name] = {"sha256": file_sha256(out / name), "shape": list(X.shape)}
    meta = {
        "started_utc": started, "finished_utc": utc_now(), "code_commit": code_commit(),
        "settings": {"bank_prompts_sha256": prompts_sha256(records), "split": args.split, "layers": layers,
                     "positions": [HORIZON], "limit": args.limit, "thinking": args.thinking,
                     "pass": "prompt only (chat template applied), batch 1, no generation",
                     "layer_convention": "L >= 1: residual stream after decoder block L; 0: input of block 1; "
                                         "n_layers: output of the last block, before the final norm"},
        "bank_name": args.bank.name, "run_name": None,
        "run_settings": {"model": args.model, "thinking": args.thinking},
        "model": model.describe(),
        "counts": {"selected": len(selected), "with_horizon": len(sites), "without_horizon": len(selected) - len(sites),
                   "rows": len(index), "decode_mismatches": 0,
                   "tokens_per_horizon": dict(sorted(Counter(len(s.tokens) for s, _ in sites.values()).items()))},
        "arrays": hashes,
    }
    (out / "extract.json").write_text(json.dumps(meta, indent=2) + "\n")
    log(f"wrote {out}: {meta['counts']}")
    return out


if __name__ == "__main__":
    main()
