"""E21 (EvalsBoundarySpec.md S1): thinking-off forward passes on the eval types, at the final assistant
header, the pre-label ':' and the last token of the record's cue; plus the model's stated horizon.

Per record: the chat template with an empty think block and the choice prefix prefilled (as E19), one
forward pass, hidden states at --layers (float16) at 7 positions, named in the saved `tokens`:
    "<|im_end|>" "\\n" "<|im_start|>" "assistant" "\\n"   the final assistant header; the last "\\n" is
                                                         suffix_end (analysis: index of "assistant" + 1)
    ":"                                                   prelabel, the last token of the choice prefix
    "phrase_end"                                          the last token of `cue_text` (NaN when the
                                                         record has no cue, so the arrays stay rectangular)
Types without an explicit horizon (persona_*, implied_event, no_horizon, twins included) also get a
greedy generation of at most 16 tokens after the prefill "The time horizon that applies here is":
`stated_horizon_text` and `stated_horizon_years` (None when no duration was written) in rows.json.

Writes <out>/<type>/acts_0.npz, rows.json (last, so its presence marks the type done; a rerun skips
done types) and <out>/settings.json ("thinking": "off"). Types that are not two-label "I choose:"
types (free_form*, the 4- and 8-option rois, json_io) are skipped, and the run says so.

Box run (from the laptop, in ~/SPAR/spar-planning-temporal-manifolds):
    just push
    just ssh "tmux new -d -s boundary 'bash -l /workspace/SPAR/spar-planning-temporal-manifolds/jared/pipeline/s1_data_vectors/queues/queue_boundary_off.sh'"
    just ssh "tail -n 5 /workspace/results/horizon/logs/queue_boundary_off.log"      # progress
    just sync                                                                          # then just down

Smoke on the 0.8B: python local_smoke_exp21.py
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from spar_horizon.config import Settings
from spar_horizon.durations import durations
from spar_horizon.eval_types import TYPES, build
from spar_horizon.model_io import _free_memory, _greedy, chat_messages, chat_text, label_ids, load_model
from spar_horizon.runs import code_hash, out_base, stamp

LAYERS = (10, 14, 16, 18, 22, 24, 26, 30)
HEADER = ("<|im_end|>", "\n", "<|im_start|>", "assistant", "\n")
TOKEN_NAMES = HEADER + (":", "phrase_end")
STATED_PREFILL = "The time horizon that applies here is"
STATED_TOKENS = 16
STATED_TYPES = ("persona_user", "persona_model", "implied_event", "no_horizon")   # no duration in the prompt
BATCH_SIZE = 8     # Qwen3-8B bf16 16.4 GB + 8 x 1.5k-token prompts of hidden states (0.45 GB each) in 24 GB


def label_type_skip_reason(records):
    """Why a type is not a two-label "I choose:" type, or None."""
    if any(r["answer_kind"] != "label" or len(r["labels"]) != 2 for r in records):
        return "not a two-label answer"
    if any(r["choice_prefix"] != "I choose:" for r in records):
        return "choice prefix is not 'I choose:'"
    return None


def cue_last_char(text, record):
    """Index in `text` of the last character of the record's cue, which sits in message `cue_message`.
    The messages appear in `text` in order, so each is searched for after the previous one."""
    cursor = 0
    for i, message in enumerate(chat_messages(record["prompt"])):
        content = message["content"].strip()
        start = text.index(content, cursor)
        cursor = start + len(content)
        if i == record["cue_message"]:
            return start + content.index(record["cue_text"].strip()) + len(record["cue_text"].strip()) - 1
    raise ValueError(f"{record['sample_uid']}: no message {record['cue_message']}")


def encode(tokenizer, record, prefill):
    """(token ids, offsets, text) of the thinking-off chat prompt with `prefill` after the template."""
    cfg = Settings(thinking="off", prefill=prefill)
    text = chat_text(tokenizer, record["prompt"], cfg)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return enc["input_ids"], enc["offset_mapping"], text


def record_positions(tokenizer, record, ids, offsets, text):
    """Indices into `ids` of TOKEN_NAMES; phrase_end is -1 when the record has no cue."""
    end = len(ids) - 1 - ids[::-1].index(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    header = list(range(end, end + len(HEADER)))
    found = tuple(tokenizer.decode([ids[j]]) for j in header)
    if found != HEADER:
        raise ValueError(f"{record['sample_uid']}: final header tokens are {found}, expected {HEADER}")
    prelabel = len(ids) - 1
    if tokenizer.decode([ids[prelabel]]) != ":":
        raise ValueError(f"{record['sample_uid']}: last prompt token is {tokenizer.decode([ids[prelabel]])!r}, not ':'")
    phrase_end = -1
    if record.get("cue_text"):
        last = cue_last_char(text, record)
        phrase_end = next(j for j, (a, b) in enumerate(offsets) if a <= last < b)
    return header + [prelabel, phrase_end]


def in_batches(items, batch_size, device, run):
    """Concatenated run(chunk) over consecutive chunks; after an out-of-memory error the chunk is
    retried and the rest of the run continues at two thirds of the batch size."""
    out, done = [], 0
    bar = tqdm(total=len(items), unit="prompt", mininterval=2, dynamic_ncols=True, leave=False)
    while done < len(items):
        chunk = items[done:done + batch_size]
        try:
            out.extend(run(chunk))
        except torch.OutOfMemoryError:
            if batch_size == 1:
                raise
            _free_memory(device)
            batch_size = max(1, batch_size * 2 // 3)
            print(f"out of memory at batch size {len(chunk)}; continuing at batch size {batch_size}", flush=True)
            continue
        done += len(chunk)
        bar.update(len(chunk))
    bar.close()
    return out


def forward_chunk(model, device, pad_id, chunk, layers):
    """[(acts [n_layers, 7, H] float16, label logits [2])] for a chunk of (ids, positions, label ids),
    left-padded; rope is relative and the pads are masked, so the padding does not change a row."""
    width = max(len(ids) for ids, _, _ in chunk)
    input_ids = torch.full((len(chunk), width), pad_id)
    mask = torch.zeros((len(chunk), width), dtype=torch.long)
    for i, (ids, _, _) in enumerate(chunk):
        input_ids[i, width - len(ids):] = torch.tensor(ids)
        mask[i, width - len(ids):] = 1
    shifted = torch.tensor([[p + width - len(ids) if p >= 0 else 0 for p in positions] for ids, positions, _ in chunk])
    with torch.no_grad():
        out = model(input_ids=input_ids.to(device), attention_mask=mask.to(device), output_hidden_states=True,
                    logits_to_keep=1)
    rows = torch.arange(len(chunk), device=device)[:, None]
    shifted = shifted.to(device)
    per_layer = torch.stack([out.hidden_states[l][rows, shifted] for l in layers], dim=1)   # [B, n_layers, 7, H]
    acts = per_layer.float().half().cpu().numpy()
    for i, (_, positions, _) in enumerate(chunk):
        acts[i][:, [j for j, p in enumerate(positions) if p < 0]] = np.nan
    logits = out.logits[:, -1].float()
    return [(acts[i], logits[i, list(ids_i)].cpu().numpy()) for i, (_, _, ids_i) in enumerate(chunk)]


def stated_horizons(tokenizer, model, device, records, batch_size):
    """[(raw text, years or None)] of the greedy continuation of STATED_PREFILL."""
    id_lists = [encode(tokenizer, r, STATED_PREFILL)[0] for r in records]

    def run(chunk):
        rows = _greedy(model, tokenizer, device, chunk, STATED_TOKENS, 0)
        texts = [tokenizer.decode(row[len(ids):], skip_special_tokens=True) for ids, row in zip(chunk, rows)]
        return [(text, parse_years(text)) for text in texts]

    return in_batches(id_lists, batch_size, device, run)


def parse_years(text):
    """Years of the first "<number> <unit>" in the text, or None."""
    found = durations(text)
    return found[0][0] if found else None


def run_type(type_id, records, tokenizer, model, device, layers, batch_size, type_dir):
    label_cache = {}
    items = []
    for r in records:
        ids, offsets, text = encode(tokenizer, r, r["choice_prefix"])
        key = tuple(r["labels"])
        if key not in label_cache:
            label_cache[key] = label_ids(tokenizer, key, Settings(prefill=r["choice_prefix"]))
        items.append((ids, record_positions(tokenizer, r, ids, offsets, text), label_cache[key]))
    results = in_batches(items, batch_size, device,
                         lambda chunk: forward_chunk(model, device, tokenizer.pad_token_id, chunk, layers))
    acts = np.stack([a for a, _ in results], axis=1)                          # [n_layers, n_rows, 7, H]
    rows = [dict(r, act_file="acts_0.npz", act_row=i, label_logits=logits.tolist(),
                 phrase_end_token=tokenizer.decode([ids[pos[-1]]]) if pos[-1] >= 0 else None)
            for i, (r, (ids, pos, _), (_, logits)) in enumerate(zip(records, items, results))]
    if type_id in STATED_TYPES:
        for row, (text, years) in zip(rows, stated_horizons(tokenizer, model, device, records, batch_size)):
            row.update(stated_horizon_text=text, stated_horizon_years=years)
    atomic_write(type_dir / "acts_0.npz", lambda f: np.savez(
        f, layers=np.asarray(layers, int), tokens=np.array(TOKEN_NAMES),
        label_logits=np.stack([l for _, l in results]), **{f"L{l}": a for l, a in zip(layers, acts)}))
    atomic_write(type_dir / "rows.json", lambda f: f.write(json.dumps(rows, indent=1).encode()))
    stamp(f"{type_id}: {len(rows)} rows" + (f", {sum(r['stated_horizon_years'] is not None for r in rows)} stated "
                                           "horizons parsed" if type_id in STATED_TYPES else ""))


def atomic_write(path, write):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        write(f)
    os.replace(tmp, path)


def plan(type_ids, seed, limit, run_dir):
    """[(type id, records, dir)] still to run; says which types are skipped and why."""
    todo = []
    for type_id in type_ids:
        records = build(type_id, seed)
        reason = label_type_skip_reason(records)
        if reason:
            stamp(f"{type_id}: skipped, {reason}")
        elif (run_dir / type_id / "rows.json").exists():
            stamp(f"{type_id}: done earlier, skipped")
        else:
            todo.append((type_id, records[:limit], run_dir / type_id))
    return todo


def run(cfg, type_ids, layers, run_dir, limit=None):
    run_dir = Path(run_dir)
    todo = plan(type_ids, cfg.seed, limit, run_dir)
    stamp(f"{len(todo)} types to run, {sum(len(r) for _, r, _ in todo)} prompts, into {run_dir}")
    if not todo:
        return
    tokenizer, model, device = load_model(cfg)
    layers = tuple(l for l in layers if l < model.config.get_text_config().num_hidden_layers)
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write(run_dir / "settings.json", lambda f: f.write(json.dumps(
        {**cfg.to_dict(), "thinking": "off", "layers": layers, "types": [t for t, _, _ in todo],
         "tokens": TOKEN_NAMES, "code_hash": code_hash()}, indent=1).encode()))
    for type_id, records, type_dir in todo:
        type_dir.mkdir(parents=True, exist_ok=True)
        run_type(type_id, records, tokenizer, model, device, layers, cfg.batch_size, type_dir)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=Settings.model)
    p.add_argument("--types", default="all", help="comma-separated type ids, or 'all'")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="records per type (smoke)")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--layers", default=",".join(map(str, LAYERS)))
    p.add_argument("--out", default=None, help="run dir (default: <out base>/boundary_<model>_think-off)")
    args = p.parse_args()
    cfg = Settings(model=args.model, thinking="off", prefill="I choose:", bank="full", batch_size=args.batch_size,
                   seed=args.seed)
    type_ids = list(TYPES) if args.types == "all" else args.types.split(",")
    run_dir = args.out or out_base() / f"boundary_{args.model.split('/')[-1]}_think-off"
    run(cfg, type_ids, tuple(int(x) for x in args.layers.split(",")), run_dir, args.limit)


if __name__ == "__main__":
    main()
