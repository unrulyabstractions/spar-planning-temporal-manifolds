"""E17: are the force-closed traces of a k-option bank a think-budget artefact? (28_spec_choice4_side.md)

    python -m pipeline.s2_steering.experiments.think_budget.exp17_think_budget --bank choice4 --unit 1
    python -m pipeline.s2_steering.experiments.think_budget.exp17_think_budget --bank choice4 --unit 2
    python -m pipeline.s2_steering.experiments.think_budget.exp17_think_budget --bank choice4 --unit 3 --batch-size 4 --max-chunks 3
    python -m pipeline.s2_steering.experiments.think_budget.exp17_think_budget --bank choice4 --unit 2 --limit 4 --extra-tokens 128 --out out/smoke

Reads steering/exp17_Qwen3-8B_think_budget/stage0_<bank>.{json,npz} (pipeline/s2_steering/analysis/think_budget_stage0.py: the
prompt ids and the exact first 3,072 think tokens of every force-closed trace; the box has no results/, so
the E0 generation cache is not there). Thinking on, greedy, as E0. Units:

  1  the original forced answer step again, in new batches: prompt + 3,072 think tokens + '</think>' +
     the answer pass of E0. Its labels against the original ones are the batching noise floor.
  2  each trace resumed from its 3,072 tokens for up to 3,072 more (total 6,144), then E0's close and
     answer logic: a trace that writes '</think>' in the extension closes naturally (forced False), the
     others are force-closed there.
  3  (optional) the traces still open after unit 2 continue 2,048 more (total 8,192), same logic.

The traces are shuffled with a fixed seed, stratified by regime, so any prefix of chunks is balanced.
A chunk of --chunk-size traces is saved as <run>/<bank>/unit<N>/chunk_NNN.json (rows) and .npz (the think
ids, units 2 and 3) when it is done; a rerun skips finished chunks, so a stopped run loses at most the
chunk in flight. Unit 3 chunk n holds the traces unit 2's chunk n left open.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from spar_horizon.config import Settings
from spar_horizon.model_io import (Generation, _free_memory, _greedy, _is_whitespace_token, answer_tokens,
                                   chat_text, close_ids, load_model, locate_label, prefill_ids, think_end_id)
from spar_horizon.runs import code_hash, out_base, stamp

JARED = Path(__file__).resolve().parents[4]
INPUTS_DIR = JARED / "steering" / "exp17_Qwen3-8B_think_budget"
SEED = 0
CHUNK_SIZE = 24
EXTRA_TOKENS = {1: 0, 2: 3072, 3: 2048}     # think tokens added by each unit
BASE_THINK = 3072                           # the original budget; the stage 0 ids are its think tokens


def balanced_order(regimes, seed=SEED):
    """Indices of `regimes` shuffled with a fixed seed so that every prefix holds each regime in its
    overall proportion: each regime's shuffled indices are spread evenly over [0, 1) at a random
    offset and the merged list is sorted by position."""
    rng = np.random.default_rng(seed)
    keyed = []
    for regime in sorted(set(regimes)):
        members = rng.permutation([i for i, r in enumerate(regimes) if r == regime])
        offset = rng.random()
        keyed += [((j + offset) / len(members), int(i)) for j, i in enumerate(members)]
    return [i for _, i in sorted(keyed)]


def split_ids(flat, offsets):
    return [flat[a:b].tolist() for a, b in zip(offsets[:-1], offsets[1:])]


def load_inputs(inputs_dir, bank):
    """(stage 0 record, per-row prompt ids, per-row think ids)."""
    record = json.loads((Path(inputs_dir) / f"stage0_{bank}.json").read_text())
    z = np.load(Path(inputs_dir) / f"stage0_{bank}.npz")
    prompts, thinks = (split_ids(z[f"{name}_ids"], z[f"{name}_offsets"]) for name in ("prompt", "think"))
    if not len(record["rows"]) == len(prompts) == len(thinks):
        raise ValueError(f"stage0_{bank} holds {len(record['rows'])} rows but {len(prompts)} prompt and {len(thinks)} think id lists")
    return record, prompts, thinks


def close_and_answer(tokenizer, model, device, plens, traces, cfg):
    """E0's close and answer logic (model_io._generate_rows after its think pass) on `traces`, each the
    prompt (plens[i] tokens) plus whatever think tokens were generated. A trace holding '</think>'
    closes naturally; one that does not is force-closed with the close text. Returns a Generation per trace
    whose think_len counts the think tokens from the prompt on."""
    tid, closer, pre = think_end_id(tokenizer), close_ids(tokenizer), prefill_ids(tokenizer, cfg)
    second, meta = [], []
    for plen, row in zip(plens, traces):
        resp = row[plen:].tolist()
        if tid in resp:
            k = resp.index(tid)
            tail = [tid]
            nxt = plen + k + 1
            if nxt < row.shape[0] and _is_whitespace_token(tokenizer, row[nxt]):
                tail.append(int(row[nxt]))
            else:
                tail += tokenizer.encode("\n\n", add_special_tokens=False)
            head, forced = row[:plen + k].tolist(), False
        else:
            k, head, tail, forced = len(resp), row.tolist(), closer, True
        second.append(head + tail + pre)
        meta.append((plen, k, forced, len(tail) + len(pre)))
    answered = _greedy(model, tokenizer, device, second, answer_tokens(cfg), answer_tokens(cfg))
    generations = []
    for ids, row, (plen, k, forced, pre_len) in zip(second, answered, meta):
        start = len(ids)
        lead = locate_label(tokenizer, row, start, cfg) - start
        generations.append(Generation(row, plen, start + lead, k, forced, pre_len + lead, lead))
    return generations


def extend_traces(tokenizer, model, device, heads, extra):
    """Each head (prompt + think tokens, no '</think>') continued greedily by up to `extra` tokens."""
    return _greedy(model, tokenizer, device, heads, extra, 0)


def run_batches(items, batch_size, process):
    """`process(batch)` over consecutive batches of `items`; after an out-of-memory error the batch is
    retried and the rest of the run continues at two thirds of the size, as model_io.generate_batch."""
    results, done = [], 0
    while done < len(items):
        batch = items[done:done + batch_size]
        try:
            results += process(batch)
        except torch.OutOfMemoryError:
            if batch_size == 1:
                raise
            _free_memory("cuda")
            batch_size = max(1, batch_size * 2 // 3)
            print(f"out of memory at batch size {len(batch)}; continuing at batch size {batch_size}", flush=True)
            continue
        done += len(batch)
    return results


def result_row(unit, meta, gen, tokenizer, base_len):
    """The saved row of one trace: its answer, close status and think length against stage 0's metadata."""
    answer = gen.answer_text(tokenizer)
    label = answer.strip()[:1]
    return {"unit": unit, "sample_uid": meta["sample_uid"], "label": label, "answer": answer,
            "correct": label == meta["truth_label"], "forced": bool(gen.forced), "think_len": int(gen.think_len),
            "new_think_len": int(gen.think_len) - base_len}


def unit_dir(run_dir, unit):
    return run_dir / f"unit{unit}"


def chunk_paths(directory, n):
    return directory / f"chunk_{n:03d}.npz", directory / f"chunk_{n:03d}.json"


def chunk_done(directory, n):
    """The json is renamed last, so its presence means the npz is complete."""
    return chunk_paths(directory, n)[1].exists()


def finished_chunks(directory):
    return sorted(int(p.stem.split("_")[1]) for p in directory.glob("chunk_*.json"))


def _atomic_write(path, write):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        write(f)
    os.replace(tmp, path)


def write_chunk(directory, n, unit, uids, rows, thinks, seconds):
    """Chunk `n`: the think ids first (units 2 and 3), then the json."""
    directory.mkdir(parents=True, exist_ok=True)
    npz_path, json_path = chunk_paths(directory, n)
    if thinks is not None:
        flat = np.concatenate([np.asarray(t, np.int32) for t in thinks]) if thinks else np.zeros(0, np.int32)
        offsets = np.r_[0, np.cumsum([len(t) for t in thinks])].astype(np.int64)
        _atomic_write(npz_path, lambda f: np.savez(f, think_ids=flat, offsets=offsets))
    meta = {"unit": unit, "chunk": n, "sample_uids": uids, "rows": rows, "seconds": round(seconds, 1)}
    _atomic_write(json_path, lambda f: f.write(json.dumps(meta, indent=1).encode()))


def read_chunk(directory, n):
    """(json record, {uid: think ids} or None) of a finished chunk."""
    npz_path, json_path = chunk_paths(directory, n)
    meta = json.loads(json_path.read_text())
    if not npz_path.exists():
        return meta, None
    with np.load(npz_path) as z:
        thinks = [z["think_ids"][a:b].tolist() for a, b in zip(z["offsets"][:-1], z["offsets"][1:])]
    return meta, dict(zip(meta["sample_uids"], thinks))


def check_chunk_matches(directory, n, uids):
    done = json.loads(chunk_paths(directory, n)[1].read_text())["sample_uids"]
    if done != uids:
        raise ValueError(f"{chunk_paths(directory, n)[1]} was written for other traces (another --limit or "
                         f"--chunk-size?); use another --out or delete {directory}")


def plan_chunks(order, rows, unit, run_dir, chunk_size):
    """[(chunk number, row indices)] of a unit. Units 1 and 2 cut the shuffled order; unit 3 takes, from each
    finished unit 2 chunk, the traces it left open."""
    spans = [(n, order[n * chunk_size:(n + 1) * chunk_size]) for n in range(-(-len(order) // chunk_size))]
    if unit != 3:
        return spans
    by_uid = {r["sample_uid"]: i for i, r in enumerate(rows)}
    previous = unit_dir(run_dir, 2)
    plan = []
    for n, _ in spans:
        if not chunk_done(previous, n):
            continue
        meta, _ = read_chunk(previous, n)
        plan.append((n, [by_uid[r["sample_uid"]] for r in meta["rows"] if r["forced"]]))
    return plan


def heads_for(unit, run_dir, n, indices, rows, prompts, thinks):
    """(prompt ids, think ids so far) per row index: stage 0's ids, or unit 2's for unit 3."""
    if unit != 3:
        return [(prompts[i], thinks[i]) for i in indices]
    _, saved = read_chunk(unit_dir(run_dir, 2), n)
    return [(prompts[i], saved[rows[i]["sample_uid"]]) for i in indices]


def run_chunk(unit, ctx, n, indices):
    """(rows, think ids or None) of one chunk."""
    tokenizer, model, device, cfg = ctx["tokenizer"], ctx["model"], ctx["device"], ctx["cfg"]
    pairs = heads_for(unit, ctx["run_dir"], n, indices, ctx["rows"], ctx["prompts"], ctx["thinks"])
    items = list(zip(indices, pairs))

    def process(batch):
        plens = [len(p) for _, (p, _) in batch]
        heads = [torch.tensor(p + t, device=device) for _, (p, t) in batch]
        traces = heads if unit == 1 else extend_traces(tokenizer, model, device, [h.tolist() for h in heads], ctx["extra"])
        return list(zip(batch, close_and_answer(tokenizer, model, device, plens, traces, cfg), traces))

    out_rows, out_thinks = [], []
    for ((i, (prompt, think)), gen, trace) in run_batches(items, ctx["batch_size"], process):
        out_rows.append(result_row(unit, ctx["rows"][i], gen, tokenizer, len(ctx["thinks"][i])))
        if unit != 1:
            generated = trace[len(prompt) + len(think):len(prompt) + gen.think_len].tolist()
            out_thinks.append(think + generated)
    return out_rows, (out_thinks if unit != 1 else None)


def check_ids_match_tokenizer(tokenizer, cfg, record, prompts):
    """The stored prompt ids are what this tokenizer makes of the stored prompt text (the ids the cache
    entries were generated from)."""
    row = record["rows"][0]
    ids = tokenizer.encode(chat_text(tokenizer, row["prompt"], cfg), add_special_tokens=False)
    if ids != prompts[0]:
        raise ValueError("the stage 0 prompt ids are not this tokenizer's tokenisation of the prompt text")


def run(args):
    record, prompts, thinks = load_inputs(args.inputs, args.bank)
    rows = record["rows"]
    n_total = len(rows)
    settings = {**record["settings"], "model": args.model}
    cfg = Settings(**settings)
    order = balanced_order([r["regime"] for r in rows])[:args.limit]
    extra = args.extra_tokens if args.extra_tokens is not None else EXTRA_TOKENS[args.unit]
    run_dir = out_base(args.out) / f"exp17_{args.model.split('/')[-1]}_think_budget" / args.bank
    directory = unit_dir(run_dir, args.unit)
    directory.mkdir(parents=True, exist_ok=True)
    plan = plan_chunks(order, rows, args.unit, run_dir, args.chunk_size)
    for n, indices in plan:
        if chunk_done(directory, n):
            check_chunk_matches(directory, n, [rows[i]["sample_uid"] for i in indices])
    todo = [(n, idx) for n, idx in plan if not chunk_done(directory, n)]
    if args.max_chunks is not None:
        todo = todo[:args.max_chunks]
    finished = sum(chunk_done(directory, n) for n, _ in plan)
    stamp(f"{args.bank} unit {args.unit}: {n_total} force-closed traces, {len(order)} used, {len(plan)} chunks of "
          f"{args.chunk_size}; {finished} done, {len(todo)} to run; +{extra} think tokens, batch {args.batch_size}")
    (run_dir / f"manifest_unit{args.unit}.json").write_text(json.dumps({
        "settings": cfg.to_dict(), "unit": args.unit, "extra_tokens": extra, "batch_size": args.batch_size,
        "chunk_size": args.chunk_size, "limit": args.limit, "n_traces": len(order), "code_hash": code_hash()}, indent=1))
    if not todo:
        return run_dir
    tokenizer, model, device = load_model(cfg)
    if args.model == record["model"]:
        check_ids_match_tokenizer(tokenizer, cfg, record, prompts)
    ctx = {"tokenizer": tokenizer, "model": model, "device": device, "cfg": cfg, "run_dir": run_dir, "rows": rows,
           "prompts": prompts, "thinks": thinks, "extra": extra, "batch_size": args.batch_size}
    for n, indices in tqdm(todo, desc=f"unit {args.unit} chunks", unit="chunk", dynamic_ncols=True):
        if not indices:
            write_chunk(directory, n, args.unit, [], [], [] if args.unit != 1 else None, 0.0)
            continue
        stamp(f"chunk {n}: {len(indices)} traces")
        t0 = time.time()
        chunk_rows, chunk_thinks = run_chunk(args.unit, ctx, n, indices)
        write_chunk(directory, n, args.unit, [rows[i]["sample_uid"] for i in indices], chunk_rows, chunk_thinks,
                    time.time() - t0)
        closed = sum(not r["forced"] for r in chunk_rows)
        stamp(f"chunk {n} written: {closed}/{len(chunk_rows)} closed, {time.time() - t0:.0f} s")
    return run_dir


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bank", choices=("choice4", "choice8"), required=True)
    parser.add_argument("--unit", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-8B",
                        help="a smaller model runs the plumbing (its tokenizer does not match the stored ids)")
    parser.add_argument("--inputs", default=str(INPUTS_DIR), help="folder with stage0_<bank>.{json,npz}")
    parser.add_argument("--out", default=None, help="output base dir (default: $WORKSPACE/results/horizon or out/)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--extra-tokens", type=int, default=None, help="think tokens to add (default: 0 / 3072 / 2048 by unit)")
    parser.add_argument("--limit", type=int, default=None, help="only the first N traces of the shuffled order (smoke)")
    parser.add_argument("--max-chunks", type=int, default=None, help="stop after this many new chunks")
    return parser


if __name__ == "__main__":
    run(make_parser().parse_args())
