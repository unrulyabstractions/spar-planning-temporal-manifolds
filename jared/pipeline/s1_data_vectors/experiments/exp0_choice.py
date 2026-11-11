"""E0 for the k-option choice banks: chunked, resumable, reduced activation set.

    python -m pipeline.s1_data_vectors.experiments.exp0_choice --model Qwen/Qwen3-8B --bank choice4 --batch-size 12
    python -m pipeline.s1_data_vectors.experiments.exp0_choice --bank choice4 --assemble        # no model: only answers.json + manifest
    python -m pipeline.s1_data_vectors.experiments.exp0_choice --model Qwen/Qwen3.5-0.8B --bank choice8 --limit 8 --chunk-size 4 \\
        --max-think-tokens 32 --batch-size 4 --out out/smoke                                        # the queue's smoke

Run from jared/. The box runs it through pipeline/s1_data_vectors/queues/queue_multichoice.sh.

The bank's records (already in a balanced order, so any prefix of chunks is balanced over
duration and regime) are cut into consecutive chunks of `--chunk-size`. Each chunk is generated
and extracted on its own and written to `<run dir>/chunks/chunk_NNN.npz` (activations at
`--layers`, keyed by hidden_states index, the kept layout including the `prompt` region of
`spar_horizon.extract`, and the per-prompt arrays) and `chunk_NNN.json` (the answers.json rows),
each through a temp file and a rename, the json last. A rerun skips finished chunks, so a killed run
loses at most the chunk in flight. No PC sweeps, probes or plots here: `pipeline/s3_analysis/by_label/choice_structure.py`
reads the chunks. After the last chunk (or with --assemble, for whatever chunks exist) the
chunks are joined into `answers.json` and `manifest.json` (chunks done, forced-close rate and
`multichoice_behavior` over the kept prompts of the finished chunks).
"""

import json
import os

import numpy as np
from tqdm import tqdm

from pipeline.s1_data_vectors.experiments.common import parse_args
from pipeline.s1_data_vectors.experiments.exp0_baseline import answer_rows
from spar_horizon.banks import build_records
from spar_horizon.behavior import multichoice_behavior
from spar_horizon.extract import extract
from spar_horizon.model_io import load_model
from spar_horizon.runs import Manifest, resolve_out_dir, stamp

EXP = "exp0"
LAYERS = (10, 14, 16, 18, 22, 24, 26, 30)
CHUNK_SIZE = 48


def chunk_paths(chunk_dir, n):
    return chunk_dir / f"chunk_{n:03d}.npz", chunk_dir / f"chunk_{n:03d}.json"


def chunk_done(chunk_dir, n):
    """The json is renamed last, so its presence means the npz is complete."""
    return chunk_paths(chunk_dir, n)[1].exists()


def finished_chunks(chunk_dir):
    return sorted(int(p.stem.split("_")[1]) for p in chunk_dir.glob("chunk_*.json"))


def _atomic_write(path, write):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        write(f)
    os.replace(tmp, path)


def write_chunk(chunk_dir, n, start, records, ext, layers):
    """Write chunk `n` (records[start:...] as extracted into `ext`)."""
    npz_path, json_path = chunk_paths(chunk_dir, n)
    regions = {f"region_{k}": np.asarray(v, int) for k, v in ext.regions.items()}
    arrays = dict(layers=np.asarray(layers, int), tokens=np.array(ext.tokens), n_suffix=ext.n_suffix, kept=ext.kept,
                  think_fractions=np.asarray(ext.think_fractions, float), forced_mask=ext.forced_mask,
                  think_len=ext.think_len, lead=ext.lead, keep_idx=ext.keep_idx, keep_ids=ext.keep_ids,
                  label_logits=ext.label_logits, option_order=ext.option_order, **regions,
                  **{f"L{layer}": a for layer, a in zip(layers, ext.activations)})
    _atomic_write(npz_path, lambda f: np.savez(f, **arrays))
    rows = answer_rows(records, ext.answers, ext.kept, zip(ext.forced_mask, ext.think_len, ext.think_text))
    meta = {"chunk": n, "start": start, "sample_uids": [r["sample_uid"] for r in records],
            "kept": ext.kept.tolist(), "rows": rows}
    _atomic_write(json_path, lambda f: f.write(json.dumps(meta, indent=1).encode()))


def check_chunk_matches(chunk_dir, n, records):
    uids = json.loads(chunk_paths(chunk_dir, n)[1].read_text())["sample_uids"]
    if uids != [r["sample_uid"] for r in records]:
        raise ValueError(f"{chunk_paths(chunk_dir, n)[1]} was written for other prompts (another --limit or "
                         f"--chunk-size?); use another --out or delete {chunk_dir}")


def usable_layers(model, layers):
    """`layers` that are raw decoder outputs of this model (hidden_states excludes the post-norm last)."""
    n = model.config.get_text_config().num_hidden_layers
    kept = tuple(l for l in layers if l < n)
    if kept != tuple(layers):
        print(f"model has {n} layers: dropping {sorted(set(layers) - set(kept))}")
    return kept


def run_chunks(cfg, out_dir, records, chunk_size, layers):
    chunk_dir = out_dir / "chunks"
    chunk_dir.mkdir(exist_ok=True)
    spans = [(n, n * chunk_size) for n in range(-(-len(records) // chunk_size))]
    for n, start in spans:
        if chunk_done(chunk_dir, n):
            check_chunk_matches(chunk_dir, n, records[start:start + chunk_size])
    todo = [(n, start) for n, start in spans if not chunk_done(chunk_dir, n)]
    stamp(f"{len(records)} records in {len(spans)} chunks of {chunk_size}; {len(spans) - len(todo)} done, {len(todo)} to run")
    if not todo:
        return
    tokenizer, model, device = load_model(cfg)
    layers = usable_layers(model, layers)
    for n, start in tqdm(todo, desc="chunks", unit="chunk", dynamic_ncols=True):
        chunk = records[start:start + chunk_size]
        stamp(f"chunk {n}: records {start}-{start + len(chunk) - 1}")
        ext = extract(tokenizer, model, device, [r["prompt"] for r in chunk], cfg, layers=layers, records=chunk,
                      prompt_positions=True)
        write_chunk(chunk_dir, n, start, chunk, ext, layers)
        stamp(f"chunk {n} written: {int(ext.kept.sum())}/{len(chunk)} kept, {ext.forced}/{len(chunk)} force-closed")


def assemble(cfg, out_dir, n_records, chunk_size):
    """answers.json and manifest.json from whatever chunks exist."""
    chunk_dir = out_dir / "chunks"
    done = finished_chunks(chunk_dir) if chunk_dir.exists() else []
    rows, label_logits = [], []
    for n in done:
        npz_path, json_path = chunk_paths(chunk_dir, n)
        meta = json.loads(json_path.read_text())
        rows += meta["rows"]
        with np.load(npz_path) as z:
            label_logits.append(z["label_logits"])
    kept = [r for r in rows if "label" in r]
    (out_dir / "answers.json").write_text(json.dumps(rows, indent=1))
    manifest = Manifest(EXP, cfg, out_dir)
    manifest.add(chunk_size=chunk_size, n_records=n_records, n_chunks_total=-(-n_records // chunk_size),
                 n_chunks_done=len(done), chunks_done=done, n_prompts=len(rows), n_kept=len(kept))
    if kept:
        manifest.add(forced=sum(r["forced"] for r in kept), forced_rate=float(np.mean([r["forced"] for r in kept])),
                     multichoice_behavior=multichoice_behavior(kept, [r["answer"] for r in kept],
                                                              np.concatenate(label_logits)))
    else:
        print("no finished chunks: answers.json is empty")
    manifest.write()


def run(cfg, out=None, chunk_size=CHUNK_SIZE, layers=LAYERS, assemble_only=False):
    out_dir = resolve_out_dir(EXP, cfg, out)
    records = build_records(cfg)
    print(f"{len(records)} prompts ({cfg.bank} bank), model {cfg.model}, thinking {cfg.thinking}")
    if not assemble_only:
        run_chunks(cfg, out_dir, records, chunk_size, layers)
    assemble(cfg, out_dir, len(records), chunk_size)
    return out_dir


if __name__ == "__main__":
    def add_args(p):
        p.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="prompts per chunk file")
        p.add_argument("--layers", default=",".join(map(str, LAYERS)), help="hidden_states indices to keep")
        p.add_argument("--assemble", action="store_true", help="only join the finished chunks; load no model")

    args, cfg = parse_args(__doc__, add_args)
    run(cfg, args.out, args.chunk_size, tuple(int(l) for l in args.layers.split(",")), args.assemble)
