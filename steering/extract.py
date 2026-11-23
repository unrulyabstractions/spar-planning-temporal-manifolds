"""Stage 1 of 2: read the residual stream at the answer label of every reply in a baseline run.

    python extract.py --bank ../prompt_bank/data/t0_s0 \\
                      --run ../prompt_bank/runs/gemma-4-31B-it__t0_s0 --layers 38
    python extract.py --bank ../prompt_bank/data/t0_s0 --run <run dir> --layers 12 15 18 --out activations/sweep

Nothing is generated here. The baseline run (../prompt_bank/baseline.py) already holds each reply's exact
generated token ids; for each prompt this script rebuilds the sequence the model saw and wrote,

    chat template(prompt)  +  generated ids up to and including the answer-label token,

runs it through the model once (teacher forcing, batch 1, so no padding), and keeps the residual stream at the
requested layers (--layers all: every layer a hook can steer, 1 .. n_layers - 1) at up to four sites (see
steer/sites.py and steer/layers.py for the exact definitions):

    label        the token carrying the answer letter
    prelabel     the token before it, whose state predicts the letter
    prompt_end   the last token of the model input, whose state predicts the first generated token
    think_end    the end-of-thinking token, whose state predicts the first token of the answer (thinking runs only)

It also keeps, at the prelabel site, the --top-logits (default 64) largest next-token logits with their token ids
(prelabel_top_logits.npz): the unsteered scores of the answer letters, among whatever else ranks that high.

The replayed pass is checked against the generation, at the token before the label
(`replay_consistent` in index.jsonl). What "consistent" means depends on how the baseline run decoded:

* greedy: the label the model wrote should be the top prediction (`replay_label_margin` >= 0). Near-ties can come
  out a bf16 step negative; a clearly negative margin means the replay is not reproducing the generation, and
  extract.json reports the smallest margin so that case cannot hide in a count.
* sampled: the label need not be the top prediction, so the margin can be honestly negative. What must hold is
  that the label was a token sampling could draw: among the run's top-k tokens (`replay_label_rank` < top_k; rank 0
  is the top prediction). This is a necessary condition, weaker than the greedy check: it does not test top-p, and
  it cannot tell which of the allowed tokens the generation drew. extract.json reports the largest rank.

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
from pbank.provenance import utc_now
from pbank.records import load_prompts, prompts_sha256, read_jsonl, write_jsonl
from steer.activations import POSITIONS, TOP_LOGITS_FILE, array_name, file_sha256
from steer.provenance import code_commit
from steer.layers import capture, check_layer, n_layers
from steer.sites import locate_label, replay_sites

HERE = Path(__file__).resolve().parent


def latest_rows(path: Path) -> dict[str, dict]:
    """The last row per uid of a responses.jsonl (a resumed run may have appended a uid twice)."""
    return {row["uid"]: row for row in read_jsonl(path)}


def forward_at(model, ids: list[int], layers: list[int], positions: list[int] | None = None,
               top: int = 0) -> tuple[dict, float, int, tuple]:
    """Run `ids` through the model. Returns ({layer: [len(positions), d] float32 array}, label margin, label rank,
    (top token ids, their logits)). `positions` are indices into `ids` (default: the last two, prelabel then label).

    Margin: the logit of the last token (the label the model generated) minus the largest logit of any other token,
    both read at the position before it. Rank: how many tokens have a larger logit there (0 = the top prediction).
    A greedy generation implies margin >= 0; the replay can come out slightly negative on a near-tie, because the
    logits are bf16 (steps of 0.125 at their usual size) and a batch-1 pass does not round exactly like the batched
    generation did. The top `top` logits are read at that same position, largest first."""
    import torch

    positions = [len(ids) - 2, len(ids) - 1] if positions is None else positions
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
    rank = int((logits > logits[label]).sum())
    best = torch.topk(logits, top) if top else None
    top_ids = best.indices.cpu().numpy().astype(np.int32) if top else np.zeros(0, np.int32)
    top_logits = best.values.cpu().numpy().astype(np.float32) if top else np.zeros(0, np.float32)
    return ({layer: acts[layer][0, positions].float().cpu().numpy() for layer in layers}, margin, rank,
            (top_ids, top_logits))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bank", type=Path, required=True, help="bank directory (../prompt_bank/data/<bank>)")
    p.add_argument("--run", type=Path, required=True, help="baseline run directory (../prompt_bank/runs/<run>)")
    p.add_argument("--layers", nargs="+", required=True,
                   help="layers to read, in the hidden_states convention (steer/layers.py), or `all` for every "
                        "layer a hook can steer (1 .. n_layers - 1); the report uses 38 for Gemma 4 31B")
    p.add_argument("--top-logits", type=int, default=64, metavar="K",
                   help="keep the K largest next-token logits at the prelabel site (0: none)")
    p.add_argument("--split", default="extraction", choices=["extraction", "verification", "all"])
    p.add_argument("--out", type=Path, help="output directory (default: activations/<run name>)")
    p.add_argument("--limit", type=int, help="only the first N prompts (smoke tests)")
    p.add_argument("--allow-offload", action="store_true",
                   help="smoke tests only: let weights that don't fit on the GPUs stay in CPU RAM (recorded)")
    args = p.parse_args()
    if args.layers != ["all"] and not all(v.isdigit() for v in args.layers):
        p.error("--layers takes layer numbers, or the single word all")
    if args.top_logits < 0:
        p.error("--top-logits must be 0 or more")

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
    decoding = settings.get("decoding", "greedy")                  # "greedy", or the sampling values (a dict)
    top_k = None if decoding == "greedy" else decoding["top_k"]
    then, now = run_meta["model"].get("model_revision"), model_revision(model_id)
    if then != now:
        sys.exit(f"the cached weights of {model_id} ({now}) are not those the run used ({then})")

    selected = [r for r in records if args.split == "all" or r.split == args.split][:args.limit]
    model = LocalChatModel(model_id, allow_offload=args.allow_offload)
    layers = list(range(1, n_layers(model.model))) if args.layers == ["all"] else sorted({int(v) for v in args.layers})
    for layer in layers:
        check_layer(model.model, layer)
    positions = [pos for pos in POSITIONS if thinking or pos != "think_end"]
    started = utc_now()
    log(f"{model_id}, thinking {'on' if thinking else 'off'}: {len(selected)} prompts of {args.bank.name}, "
        f"layers {layers[0]}..{layers[-1]} ({len(layers)}), sites {list(positions)}; baseline decoding {decoding}")

    kept, skipped = [], []
    arrays = {(layer, pos): [] for layer in layers for pos in positions}
    top_ids, top_logits = [], []
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
        where = replay_sites(len(prompt_ids), site, thinking)        # site -> index in `ids`
        acts, margin, rank, top = forward_at(model, ids, layers, [where[pos] for pos in positions], args.top_logits)
        for layer in layers:
            for k, pos in enumerate(positions):
                arrays[(layer, pos)].append(acts[layer][k])
        top_ids.append(top[0]), top_logits.append(top[1])
        kept.append({"row": len(kept), "uid": rec.uid, "choice_label": row["choice_label"],
                     "chosen_option": row["chosen_option"],
                     "label_token": model.tok.decode([ids[-1]]), "prelabel_token": model.tok.decode([ids[-2]]),
                     "label_index": site.label_index, "seq_len": len(ids),
                     "prompt_len": len(prompt_ids), "answer_start": site.answer_start,
                     **{f"{pos}_token": model.tok.decode([ids[where[pos]]]) for pos in positions
                        if pos not in ("label", "prelabel")},
                     "replay_label_margin": round(margin, 4), "replay_predicts_label": margin >= 0,
                     "replay_label_rank": rank,
                     "replay_consistent": margin >= 0 if top_k is None else rank < top_k})
        if n % 25 == 0 or n == len(selected):
            log(f"  [{n}/{len(selected)}] kept {len(kept)}, skipped {len(skipped)}, "
                f"replay mismatches {sum(not k['replay_consistent'] for k in kept)}")

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
    top_meta = None
    if args.top_logits:
        np.savez(out / TOP_LOGITS_FILE, token_ids=np.stack(top_ids), logits=np.stack(top_logits))
        top_meta = {"file": TOP_LOGITS_FILE, "sha256": file_sha256(out / TOP_LOGITS_FILE), "k": args.top_logits,
                    "shape": [len(top_ids), args.top_logits],
                    "what": "at the prelabel site of the replayed pass: the k largest next-token logits "
                            "(float32, from bf16), largest first, and their token ids; rows as index.jsonl"}
    meta = {
        "started_utc": started, "finished_utc": utc_now(), "code_commit": code_commit(),
        "settings": {"bank_prompts_sha256": bank_sha, "split": args.split, "layers": layers,
                     "positions": positions, "limit": args.limit, "replay": "teacher-forced, batch 1, bf16"},
        "bank_name": args.bank.name, "run_name": args.run.name, "run_settings": settings,
        "model": model.describe(),
        "counts": {"selected": len(selected), "kept": len(kept), "skipped": len(skipped),
                   "skipped_by_reason": {r: sum(s["reason"] == r for s in skipped) for r in {s["reason"] for s in skipped}},
                   "replay_check": "greedy: label is the top prediction" if top_k is None else
                                   f"sampled: label is among the top {top_k} predictions",
                   "replay_mismatches": sum(not k["replay_consistent"] for k in kept),
                   "replay_label_not_top": sum(not k["replay_predicts_label"] for k in kept),
                   "replay_min_label_margin": min(k["replay_label_margin"] for k in kept),
                   "replay_max_label_rank": max(k["replay_label_rank"] for k in kept)},
        "arrays": hashes, "prelabel_top_logits": top_meta,
    }
    (out / "extract.json").write_text(json.dumps(meta, indent=2) + "\n")
    log(f"wrote {out}: {meta['counts']}")


if __name__ == "__main__":
    main()
