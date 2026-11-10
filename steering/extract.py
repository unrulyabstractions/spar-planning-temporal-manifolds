"""Stage 1 of 2: read the residual stream at the answer label of every reply in a baseline run.

    python extract.py --bank ../prompt_bank/data/t0_s0 \\
                      --run ../prompt_bank/runs/gemma-4-31B-it__t0_s0 --layers 38
    python extract.py --bank ../prompt_bank/data/t0_s0 --run <run dir> --layers 12 15 18 --out activations/sweep

Nothing is generated here. The baseline run (../prompt_bank/baseline.py) already holds each reply's exact
generated token ids; for each prompt this script rebuilds the sequence the model saw and wrote,

    chat template(prompt)  +  generated ids up to and including the answer-label token,

runs it through the model once (teacher forcing, batch 1, so no padding), and keeps the residual stream at the
requested layers at two sites: the label token and the token before it (see steer/sites.py and steer/layers.py
for the exact definitions). The replayed pass is checked against the generation: at the token before the label,
the label the model actually wrote should be the top prediction (`replay_label_margin` >= 0 in index.jsonl).
Near-ties can come out a bf16 step negative; a clearly negative margin means the replay is not reproducing the
generation, and extract.json reports the smallest margin so that case cannot hide in a count.

Which prompts: those of --split (default "extraction") whose reply was parsed to a choice. Replies with no
answer (thinking never closed), a format failure or conflicting choices have no label token and are listed in
skipped.jsonl with the reason, never silently dropped. The bank's no-horizon (null) prompts are extracted too;
fit.py decides what to use.

Output: activations/<run name>/ unless --out is given; format in steer/activations.py. An existing output
directory is refused: an extraction is a record, so redo it in a new directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from pbank.local_model import LocalChatModel, log, model_revision
from pbank.provenance import git_commit, utc_now
from pbank.records import load_prompts, prompts_sha256, read_jsonl, write_jsonl
from steer.activations import POSITIONS, array_name, file_sha256
from steer.layers import capture, check_layer
from steer.sites import locate_label

HERE = Path(__file__).resolve().parent


def latest_rows(path: Path) -> dict[str, dict]:
    """The last row per uid of a responses.jsonl (a resumed run may have appended a uid twice)."""
    return {row["uid"]: row for row in read_jsonl(path)}


def forward_at(model, ids: list[int], layers: list[int]) -> tuple[dict, float]:
    """Run `ids` through the model. Returns ({layer: [2, d] float32 array of the last two positions}, label
    margin): the logit of the last token (the label the model generated) minus the largest logit of any other
    token, both read at the position before it. A greedy generation implies margin >= 0; the replay can come out
    slightly negative on a near-tie, because the logits are bf16 (steps of 0.125 at their usual size) and a
    batch-1 pass does not round exactly like the batched generation did."""
    import torch

    x = torch.tensor([ids], device=model.input_device)
    with torch.no_grad(), capture(model.model, layers) as acts:
        try:
            out = model.model(input_ids=x, use_cache=False, logits_to_keep=2)
        except TypeError:   # a model class without logits_to_keep: compute every position's logits
            out = model.model(input_ids=x, use_cache=False)
    logits = out.logits[0, -2].float()
    label = ids[-1]
    others = torch.cat([logits[:label], logits[label + 1:]])
    margin = float(logits[label] - others.max())
    return {layer: acts[layer][0, -2:].float().cpu().numpy() for layer in layers}, margin


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bank", type=Path, required=True, help="bank directory (../prompt_bank/data/<bank>)")
    p.add_argument("--run", type=Path, required=True, help="baseline run directory (../prompt_bank/runs/<run>)")
    p.add_argument("--layers", type=int, nargs="+", required=True,
                   help="layers to read, in the hidden_states convention (steer/layers.py); the report uses 38 "
                        "for Gemma 4 31B")
    p.add_argument("--split", default="extraction", choices=["extraction", "verification", "all"])
    p.add_argument("--out", type=Path, help="output directory (default: activations/<run name>)")
    p.add_argument("--limit", type=int, help="only the first N prompts (smoke tests)")
    args = p.parse_args()

    records = load_prompts(args.bank)
    bank_sha = prompts_sha256(records)
    run_meta = json.loads((args.run / "run.json").read_text())
    settings = run_meta["settings"]
    if settings["bank_prompts_sha256"] != bank_sha:
        sys.exit(f"{args.run} was run on a different bank (prompts SHA-256 {settings['bank_prompts_sha256'][:12]}, "
                 f"{args.bank} has {bank_sha[:12]})")
    rows = latest_rows(args.run / "responses.jsonl")
    if any("generated_ids" not in r or r["generated_ids"] is None for r in rows.values()):
        sys.exit(f"{args.run} has replies without generated_ids; rerun the baseline with the current baseline.py")

    out = args.out or HERE / "activations" / args.run.name
    if out.exists():
        sys.exit(f"{out} exists; extractions are records, so write a new one with --out")

    model_id, thinking = settings["model"], settings["thinking"]
    then, now = run_meta["model"].get("model_revision"), model_revision(model_id)
    if then != now:
        sys.exit(f"the cached weights of {model_id} ({now}) are not those the run used ({then})")

    selected = [r for r in records if args.split == "all" or r.split == args.split][:args.limit]
    model = LocalChatModel(model_id)
    layers = sorted(set(args.layers))
    for layer in layers:
        check_layer(model.model, layer)
    started = utc_now()
    log(f"{model_id}, thinking {'on' if thinking else 'off'}: {len(selected)} prompts of {args.bank.name}, "
        f"layers {layers}")

    kept, skipped = [], []
    arrays = {(layer, pos): [] for layer in layers for pos in POSITIONS}
    for n, rec in enumerate(selected, 1):
        row = rows.get(rec.uid)
        if row is None:
            skipped.append({"uid": rec.uid, "reason": "not in run"})
            continue
        if row["parse_status"] != "ok":
            skipped.append({"uid": rec.uid, "reason": row["parse_status"]})
            continue
        site = locate_label(model.tok, row["generated_ids"], tuple(rec.labels), thinking)
        if site is None or site.label_char.lower() != row["choice_label"].rstrip(")").lower():
            # The parser accepted this reply, so the locator must find the same label; anything else is a bug.
            raise AssertionError(f"{rec.uid}: label site {site} disagrees with parsed choice {row['choice_label']}")

        prompt_ids = model.tok(model.chat_text(rec.text, thinking), add_special_tokens=False)["input_ids"]
        ids = prompt_ids + row["generated_ids"][:site.label_index + 1]
        acts, margin = forward_at(model, ids, layers)
        for layer in layers:
            arrays[(layer, "prelabel")].append(acts[layer][0])
            arrays[(layer, "label")].append(acts[layer][1])
        kept.append({"row": len(kept), "uid": rec.uid, "choice_label": row["choice_label"],
                     "chosen_option": row["chosen_option"],
                     "label_token": model.tok.decode([ids[-1]]), "prelabel_token": model.tok.decode([ids[-2]]),
                     "label_index": site.label_index, "seq_len": len(ids),
                     "replay_label_margin": round(margin, 4), "replay_predicts_label": margin >= 0})
        if n % 25 == 0 or n == len(selected):
            log(f"  [{n}/{len(selected)}] kept {len(kept)}, skipped {len(skipped)}, "
                f"replay mismatches {sum(not k['replay_predicts_label'] for k in kept)}")

    if not kept:
        sys.exit(f"no reply could be extracted; skipped: {skipped[:5]}")
    out.mkdir(parents=True)
    write_jsonl(out / "index.jsonl", kept)
    write_jsonl(out / "skipped.jsonl", skipped)
    hashes = {}
    for (layer, pos), vectors in arrays.items():
        name = array_name(layer, pos)
        np.save(out / name, np.stack(vectors).astype(np.float32))
        hashes[name] = {"sha256": file_sha256(out / name), "shape": [len(vectors), int(vectors[0].shape[0])]}
    meta = {
        "started_utc": started, "finished_utc": utc_now(), "code_commit": git_commit(),
        "settings": {"bank_prompts_sha256": bank_sha, "split": args.split, "layers": layers,
                     "positions": list(POSITIONS), "limit": args.limit, "replay": "teacher-forced, batch 1, bf16"},
        "bank_name": args.bank.name, "run_name": args.run.name, "run_settings": settings,
        "model": model.describe(),
        "counts": {"selected": len(selected), "kept": len(kept), "skipped": len(skipped),
                   "skipped_by_reason": {r: sum(s["reason"] == r for s in skipped) for r in {s["reason"] for s in skipped}},
                   "replay_mismatches": sum(not k["replay_predicts_label"] for k in kept),
                   "replay_min_label_margin": min(k["replay_label_margin"] for k in kept)},
        "arrays": hashes,
    }
    (out / "extract.json").write_text(json.dumps(meta, indent=2) + "\n")
    log(f"wrote {out}: {meta['counts']}")


if __name__ == "__main__":
    main()
