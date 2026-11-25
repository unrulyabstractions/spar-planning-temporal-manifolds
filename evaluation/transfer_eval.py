"""Transfer to the eval types (evaluation_design §9, stage 6, on the replay readout): does a method that steers the
horizon on T0 also steer it on realistic variations of the T0 prompts?

The methods, nulls and strength are T0's: the same bundles, shared draws and shuffled refits as replay_eval.py, at the
x frozen by its calibrate stage (calibration.json of the T0 run), never recalibrated here. The prompts are Jared's eval
types, one bank per type, exported by jared/pipeline/s1_data_vectors/analysis/export_evaltype_banks.py (tier
"evaltype", split "transfer", which no fitting code selects; the eval-type record in `notes`).

Stages, per type (each resumable; outputs in --out/<type_id>/):
  baseline  the unsteered reply to every near_only / both prompt, with the T0 run's model, thinking mode, cap and batch
            size (run/run.json, run/responses.jsonl in prompt_bank/baseline.py's format). Not baseline.py itself: it
            sends `text` as one user turn, and a conversation-rendered type (system message, earlier turns) must be
            rendered from notes["messages"].
  items     replay_eval's items stage on that run (skipped.jsonl lists the prompts that cannot be read out)
  replay    replay_eval's replay stage on the type's items: every arm at +x and -x, the unsteered pass, the N0 check,
            and the prompting baseline (replay_eval.PROMPTS: the disposition and shift sentence pairs and the neutral
            sentence, appended to the last user turn and read with the reasoning held fixed, like the edits)
Once, over all types:
  report    analyze.py's per-arm statistics and null comparisons, per type and group (the type's own prompts; its
            explicit twins, the same options in the T0 template), beside the same numbers on T0's verification grid
            -> transfer_report.json, transfer_report.md

What to run is chosen by a check set (`--set transfer_sets/<name>.json`, run_local.sh: TRANSFER_SET), so a pilot on a
few hundred prompts and the full run differ by one file. Its keys, all optional except `types`:
  types           {type_id: expectation}, e.g. "expected to work"; the expectation is printed beside the type's results
  regimes         prompt regimes to run (default near_only and both, the only ones with a readout)
  variants        eval-type variants: base, swap (the reversed option order), twin (the T0 template); default all
  rows_per_type   at most this many rows per type, taken as whole items (an item's base, swap and twin together, so order
                  pairs stay complete) in bank order, which alternates near and both items
  arms            arm names, families or methods to replay (e.g. "caa", "curve-displace", "isotropic",
                  "shuffled:caa", "prompting"); default every arm, the prompting baseline included
  k               draws per null family (overrides --k)
Outputs go to --out/<set name>/, so two sets never share a baseline run. Without --set, every near_only / both row of
every bank (or of --types) runs, with every arm.

Usage: see run_local.sh (TRANSFER=1).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from functools import partial
from pathlib import Path

import numpy as np

import analyze
import replay_eval as rv

SWITCH_MIN = 5          # readable items needed with each correct option before a switch point is fitted
STATS = ("meaning", "odd_mean", "delta_log10_H")
SWITCH_KEYS = ("delta_log10_H", "switch_loc_plus", "switch_loc_minus", "slope_plus", "slope_minus",
               "plateau_low_plus", "plateau_high_plus", "plateau_low_minus", "plateau_high_minus")
GROUPS = {"prompts": ("base", "swap"), "twins": ("twin",)}
SET_KEYS = {"description", "types", "regimes", "variants", "rows_per_type", "arms", "k"}


# ------------------------------------------------------------------ check sets

def load_set(path: Path) -> dict:
    """A check set (module docstring); refuses unknown keys, so a typo cannot silently run everything."""
    sel = json.loads(Path(path).read_text())
    unknown = set(sel) - SET_KEYS
    if unknown:
        sys.exit(f"{path}: unknown keys {sorted(unknown)}; allowed: {sorted(SET_KEYS)}")
    if not isinstance(sel.get("types"), dict) or not sel["types"]:
        sys.exit(f"{path}: `types` must map each type_id to its expectation")
    bad = set(sel.get("regimes", ())) - set(rv.SCORED)
    if bad:
        sys.exit(f"{path}: regimes {sorted(bad)} have no readout; choose from {list(rv.SCORED)}")
    return sel


def select_rows(rows: list[dict], sel: dict) -> list[dict]:
    """The bank rows a check set runs: its regimes and variants, then whole items in bank order up to rows_per_type."""
    regimes = sel.get("regimes", rv.SCORED)
    variants = sel.get("variants", ("base", "swap", "twin"))
    rows = [r for r in rows if r["regime"] in regimes and r["notes"]["variant"] in variants]
    limit = sel.get("rows_per_type")
    if limit is None:
        return rows
    items = {}
    for r in rows:
        items.setdefault(r["notes"]["twin_id"], []).append(r)
    chosen = []
    for item_rows in items.values():
        if len(chosen) + len(item_rows) > limit:
            break
        chosen += item_rows
    return chosen


# ------------------------------------------------------------------ model input

def model_input(row: dict):
    """What the model is given for a bank row: the conversation when the row has one, else its text."""
    return (row.get("notes") or {}).get("messages") or row["text"]


def chat_text(tok, prompt, thinking: bool) -> str:
    """pbank's LocalChatModel.chat_text, for a user text or a whole conversation (list of messages)."""
    messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)


def load_model(model_id: str) -> rv.Model:
    M = rv.Model(model_id)
    # LocalChatModel.generate and Model.prompt_ids both go through lcm.chat_text, which takes one user string;
    # replacing it on this instance lets both take conversations without touching pbank.
    M.lcm.chat_text = partial(chat_text, M.lcm.tok)
    return M


class TransferReplayer(rv.Replayer):
    """replay_eval's Replayer, with the prompt rendered from model_input (conversations included)."""

    def prompt_of(self, it):
        return model_input(self.bank[it["uid"]])


# ------------------------------------------------------------------ baseline

def inputs_sha256(rows: list[dict]) -> str:
    """SHA-256 over uid and model input: unlike pbank's prompts hash, it covers a conversation's every turn."""
    h = hashlib.sha256()
    for r in rows:
        h.update(f"{r['uid']}|{json.dumps(model_input(r), ensure_ascii=False)}\n".encode())
    return h.hexdigest()


class ConversationSeeds:
    """A pbank Decoding whose per-batch seed also takes conversations (lists of messages), which pbank's
    batch_seed cannot hash; a user text gets the same seed as in pbank. The T0 run's decoding is followed, so a
    sampled T0 baseline (thinking on) gets a sampled baseline here: greedy decoding can loop in thinking mode."""

    def __init__(self, decoding):
        self._decoding = decoding

    def __getattr__(self, name):
        return getattr(self._decoding, name)

    def batch_seed(self, prompts) -> int:
        return self._decoding.batch_seed([p if isinstance(p, str) else json.dumps(p, ensure_ascii=False)
                                          for p in prompts])


def stage_baseline(bank_dir: Path, run_dir: Path, lcm, settings: dict, sel: dict | None = None) -> None:
    """Unsteered replies to the bank rows the check set selects (default: every near_only / both prompt); resumes,
    refuses a run with other settings."""
    from pbank.answers import parse_choice, score
    from pbank.records import append_jsonl, record_from_dict
    rows = select_rows(rv.read_jsonl(bank_dir / "prompts.jsonl"), sel or {})
    run_settings = {"model": settings["model_id"], "bank_inputs_sha256": inputs_sha256(rows),
                    "thinking": settings["thinking"], "max_new_tokens": settings["max_new_tokens"],
                    "batch_size": settings["batch_size"], "decoding": settings.get("decoding", "greedy"),
                    "prompt": "notes.messages when present, else text", "n_rows": len(rows),
                    "selection": {k: v for k, v in (sel or {}).items() if k in ("regimes", "variants", "rows_per_type")}}
    run_json = run_dir / "run.json"
    if run_json.exists():
        previous = json.loads(run_json.read_text())["settings"]
        if previous != run_settings:
            sys.exit(f"{run_dir} holds a run with other settings:\n  {previous}\nvs now\n  {run_settings}")
    responses = run_dir / "responses.jsonl"
    done = {r["uid"] for r in rv.read_jsonl(responses)} if responses.exists() else set()
    todo = [r for r in rows if r["uid"] not in done]
    if not todo:
        print(f"baseline {bank_dir.name}: {len(rows)} done, skipping")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    rv.write_json(run_json, {"settings": run_settings, "bank": str(bank_dir)})
    decoding = ConversationSeeds(rv.gen_decodings(run_settings["decoding"], 1)[0])
    bs = settings["batch_size"]
    for i in range(0, len(todo), bs):
        batch = todo[i:i + bs]
        replies = lcm.generate([model_input(r) for r in batch], thinking=settings["thinking"],
                               max_new_tokens=settings["max_new_tokens"], decoding=decoding)
        out = []
        for row, reply in zip(batch, replies):
            rec = record_from_dict(row)
            parsed = parse_choice(reply.answer, rec.labels)
            s = score(rec, parsed)
            out.append({"uid": rec.uid, "model": settings["model_id"], "thinking": reply.thinking,
                        "answer": reply.answer, "choice_label": parsed.label, "parse_status": parsed.status,
                        "n_choice_mentions": parsed.n_mentions, "chosen_option": s.chosen_option,
                        "correct": s.correct, "n_generated": reply.n_generated,
                        "hit_token_cap": reply.hit_token_cap, "seconds": round(reply.seconds, 2),
                        "generated_ids": reply.generated_ids})
        append_jsonl(responses, out)                         # appended per batch, as baseline.py: a rerun resumes
        print(f"baseline {bank_dir.name}: [{len(done) + i + len(batch)}/{len(rows)}] "
              f"{sum(r['parse_status'] == 'ok' for r in out)}/{len(out)} parsed", flush=True)


# ------------------------------------------------------------------ report

def group_summary(out: Path, items: list[dict], keep, y0, mass0, n_unreadable: int) -> dict:
    """Per-arm statistics and null comparisons on items[keep]; switch-point statistics only where the group has
    SWITCH_MIN readable items with each correct option (else the boundary is not in the data, e.g. no values given)."""
    kept = [it for it, k in zip(items, keep) if k]
    n_near = sum(it["correct_option"] == "near" for it in kept)
    n_far = sum(it["correct_option"] == "far" for it in kept)
    switch = n_near >= SWITCH_MIN and n_far >= SWITCH_MIN
    res = analyze.arm_results(out, items, y0, keep)
    if not switch:
        for r in res.values():
            for key in SWITCH_KEYS:
                r.pop(key, None)
    scored = [(y, it["correct_option"]) for y, it in zip(y0[keep], kept) if it["correct_option"] is not None]
    logr = np.log10([it["h_over_far_delay"] for it in kept])
    return {
        "n_items": len(kept), "n_unreadable": n_unreadable, "n_correct_near": n_near, "n_correct_far": n_far,
        "switch_applies": switch,
        "unsteered": {"label_mass": float(mass0[keep].mean()),
                      "accuracy": float(np.mean([(y > 0) == (c == "far") for y, c in scored])) if scored else None,
                      "switch_loc": analyze.fit_switch(logr, analyze.sigmoid(y0[keep]))[0] if switch else None},
        "arms": {name: {k: v for k, v in r.items() if k in ("x", "kind", "family", "meaning", "odd_mean", "even_mean",
                                                             "delta_log10_H", "label_mass_plus", "label_mass_minus")}
                 for name, r in res.items() if r["family"] in analyze.REPORTED},
        "comparisons": analyze.compare_to_nulls(res, STATS),
    }


def t0_summary(t0_out: Path) -> dict:
    items = rv.load_items(t0_out, "verification")
    base = np.load(t0_out / "arms" / "_none.npz")
    skipped = rv.read_jsonl(t0_out / "skipped.jsonl")
    keep = np.ones(len(items), bool)
    return group_summary(t0_out, items, keep, base["y"], base["mass"],
                         sum(s["split"] == "verification" for s in skipped))


def type_summary(type_out: Path, bank_dir: Path) -> dict:
    variant = {r["uid"]: r["notes"]["variant"] for r in rv.read_jsonl(bank_dir / "prompts.jsonl")}
    items = rv.load_items(type_out, "transfer")
    base = np.load(type_out / "arms" / "_none.npz")
    skipped = rv.read_jsonl(type_out / "skipped.jsonl")
    groups = {}
    for g, variants in GROUPS.items():
        keep = np.array([variant[it["uid"]] in variants for it in items], bool)
        if keep.any():
            n_unreadable = sum(variant[s["uid"]] in variants for s in skipped)
            groups[g] = group_summary(type_out, items, keep, base["y"], base["mass"], n_unreadable)
    return groups


def fmt(v, spec="+.3f"):
    return "–" if v is None or not np.isfinite(v) else format(v, spec)


def report_md(rep: dict) -> str:
    L = ["# Transfer to the eval types (replay readout)", "",
         f"- Strength frozen on T0 ({rep['frozen']['from']}): push x = {rep['frozen']['x_push']:g} × typical norm, "
         f"target shift = {rep['frozen']['x_shift']:g} decades",
         "- Groups: `prompts` = the type's own prompts (base and swap), `twins` = the same options in the T0 template "
         "with the horizon stated. Only near_only / both prompts are run.",
         "- meaning: order-balanced effect on the base/swap pairs (40 per type); odd: mean odd part over all items; "
         f"Δlog10 H*: switch-point shift, only where ≥ {SWITCH_MIN} readable items have each correct option.",
         "- unsteered acc: share of items whose unsteered replay prefers the correct option."]
    if rep["set"]:
        L.append(f"- Check set `{rep['set']['name']}`: {rep['set'].get('description', '')}")
        L += [f"  - {t}: {why}" for t, why in rep["set"]["types"].items()]
    if rep["skipped_types"]:
        L.append("- Not run: " + "; ".join(f"{t} ({why})" for t, why in rep["skipped_types"].items()))
    expect = rep["set"]["types"] if rep["set"] else {}
    rows = [("T0 verification grid", "grid", rep["t0"])] + [
        (f"{t} ({expect[t]})" if t in expect else t, g, s) for t, groups in rep["types"].items() for g, s in groups.items()]
    L += ["", "## Summary", "",
          "| bank | group | n (unreadable) | unsteered mass / acc | method | meaning | odd | Δlog10 H* | "
          "label mass ± | max p over nulls (meaning / odd) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for bank, g, s in rows:
        u = s["unsteered"]
        for name, a in s["arms"].items():
            if name == "curve-snap@0":
                continue
            comps = s["comparisons"].get(name, {})
            maxp = [max((c[stat]["p"] for c in comps.values() if stat in c), default=None)
                    for stat in ("meaning", "odd_mean")]
            L.append(f"| {bank} | {g} | {s['n_items']} ({s['n_unreadable']}) | {fmt(u['label_mass'], '.3f')} / "
                     f"{fmt(u['accuracy'], '.2f')} | {name} | {fmt(a.get('meaning'))} | {fmt(a.get('odd_mean'))} | "
                     f"{fmt(a.get('delta_log10_H'))} | {fmt(a.get('label_mass_plus'), '.3f')} / "
                     f"{fmt(a.get('label_mass_minus'), '.3f')} | {fmt(maxp[0], '.2f')} / {fmt(maxp[1], '.2f')} |")
    L += ["", "## Against the nulls (rank p; |null| 95th percentile)", "",
          "| bank | group | method | null family | K | meaning: stat / p95 / p | odd: stat / p95 / p | "
          "Δlog10 H*: stat / p95 / p |", "|---|---|---|---|---|---|---|---|"]
    for bank, g, s in rows:
        for name, by in s["comparisons"].items():
            for fam, st in by.items():
                cells = []
                for stat in STATS:
                    c = st.get(stat, {})
                    cells.append(f"{fmt(c.get('stat'))} / {fmt(c.get('null_p95_abs'), '.3f')} / {fmt(c.get('p'), '.2f')}")
                K = next(c["K"] for c in st.values())
                L.append(f"| {bank} | {g} | {name} | {fam} | {K} | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


def stage_report(args, types: list[str], skipped_types: dict, sel: dict | None = None) -> dict:
    out = Path(args.out)
    t0_out = Path(args.t0_out)
    cal = json.loads((t0_out / "calibration.json").read_text())
    rep = {"frozen": {"from": str(t0_out / "calibration.json"), "x_push": cal["x_push"], "x_shift": cal["x_shift"]},
           "set": {"name": out.name, **sel} if sel else None,
           "t0": t0_summary(t0_out), "types": {}, "skipped_types": dict(skipped_types)}
    for t in types:
        if t in skipped_types:
            continue
        if not (out / t / "arms" / "_none.npz").exists():
            rep["skipped_types"][t] = "no replay results"
            continue
        rep["types"][t] = type_summary(out / t, Path(args.banks) / t)
    rv.write_json(out / "transfer_report.json", rep)
    md = report_md(rep)
    tmp = out / "transfer_report.md.tmp"
    tmp.write_text(md)
    tmp.replace(out / "transfer_report.md")
    print(md)
    return rep


# ------------------------------------------------------------------ driver

def type_args(args, t: str) -> tuple[argparse.Namespace, argparse.Namespace]:
    """The namespaces replay_eval's stages read: items on the type's bank and run; replay on T0's bank (the
    shuffled-label refits draw horizons from T0's fit prompts) with the type's output directory."""
    type_out = Path(args.out) / t
    items_ns = argparse.Namespace(bank=str(Path(args.banks) / t), run=str(type_out / "run"), out=str(type_out),
                                  extraction_index=None)
    replay_ns = argparse.Namespace(bank=args.t0_bank, activations=args.activations, methods=args.methods,
                                   draws=args.draws, k=args.k, n0_check=args.n0_check, out=str(type_out),
                                   site=args.site, prompting=True)   # the prompting baseline runs here by default
    return items_ns, replay_ns


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("stages", nargs="+", choices=["baseline", "items", "replay", "report"])
    p.add_argument("--banks", required=True, help="export_evaltype_banks.py's output: one <type_id>/ bank per type")
    p.add_argument("--types", nargs="+", help="default: every bank under --banks (not with --set)")
    p.add_argument("--set", help="a check set, transfer_sets/<name>.json (module docstring); outputs go to --out/<name>/")
    p.add_argument("--t0-bank", required=True), p.add_argument("--t0-run", required=True)
    p.add_argument("--t0-out", required=True, help="replay_eval.py's output on T0 (calibration.json, items, arms)")
    p.add_argument("--activations", required=True), p.add_argument("--methods", required=True)
    p.add_argument("--draws", required=True), p.add_argument("--out", required=True)
    p.add_argument("--site", help="use only bundles fitted at this site, as replay_eval.py --site")
    p.add_argument("--k", type=int, default=rv.sd.K_GEN, help="draws per null family (rows [:k] of the shared draws)")
    p.add_argument("--n0-check", type=int, default=8)
    p.add_argument("--replay-batch", type=int, default=16, help="prompts per teacher-forced replay pass")
    p.add_argument("--replay-dtype", default="float32", choices=["float32", "bfloat16"],
                   help="weights during replay (replay_eval.weights_in)")
    args = p.parse_args(argv)
    sel = {}
    if args.set:
        if args.types:
            sys.exit("--types and --set: list the types in the set instead")
        sel = load_set(Path(args.set))
        args.out = str(Path(args.out) / Path(args.set).stem)
        args.k = sel.get("k", args.k)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    banks = Path(args.banks)
    types = list(sel["types"]) if sel else (
        args.types or sorted(d.name for d in banks.iterdir() if (d / "prompts.jsonl").exists()))
    skipped_types = {t: "no bank" for t in types if not (banks / t / "prompts.jsonl").exists()}
    cal_path = Path(args.t0_out) / "calibration.json"
    if not cal_path.exists():
        sys.exit(f"{cal_path} missing: run replay_eval.py calibrate on T0 first (the strength is frozen there)")
    cal = json.loads(cal_path.read_text())
    settings = rv.run_settings(Path(args.t0_run))
    rv.write_json(out / "settings.json", {"args": vars(args), "run": settings, "calibration": cal})
    model_stages = [s for s in ("baseline", "items", "replay") if s in args.stages]
    if model_stages:
        M = load_model(settings["model_id"])
        from steer.methods import load
        layer = load(rv.method_dirs(Path(args.methods), args.site)["caa"]).layer
        for t in types:
            if t in skipped_types:
                continue
            items_ns, replay_ns = type_args(args, t)
            Path(items_ns.out).mkdir(parents=True, exist_ok=True)
            if "baseline" in args.stages:
                stage_baseline(Path(items_ns.bank), Path(items_ns.run), M.lcm, settings, sel)
            if not (Path(items_ns.run) / "responses.jsonl").exists():
                skipped_types[t] = "no baseline run"
                continue
            if "items" in args.stages:
                rv.build_items(items_ns, M, settings)
            if "replay" in args.stages:
                rep = TransferReplayer(M, Path(items_ns.run), Path(items_ns.bank), settings["thinking"], layer,
                                       args.replay_batch)
                with rv.weights_in(M.model, args.replay_dtype):
                    rv.stage_replay(replay_ns, M, settings, rep, cal, split="transfer",
                                    keep_arm=partial(rv.arm_selected, wanted=sel.get("arms")))
    if "report" in args.stages:
        stage_report(args, types, skipped_types, sel)


if __name__ == "__main__":
    main()
