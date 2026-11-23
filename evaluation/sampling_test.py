"""Does seeded sampling fix the thinking-mode loops? Re-generate a subset of a greedy baseline run with sampling.

The greedy baseline (pbank, do_sample=False) loses many answers to repetition loops that run to the token cap
(check_loops.py). Qwen3 model cards advise against greedy decoding in thinking mode and recommend temperature 0.6,
top-p 0.95, top-k 20, min-p 0. This script re-generates, with those settings and the same cap and batch size:
  * looping prompts: cap hits of the greedy run whose tail is a repetition loop (stratified by regime), and
  * controls: prompts the greedy run answered, to see whether sampling changes the choices.
Everything else is pbank's own code (LocalChatModel, parse_choice, score); only model.generate is wrapped to sample.
Each batch is seeded from its uids, so a rerun with the same selection and batch size gives the same replies.

Usage (from evaluation/):
  python sampling_test.py --bank ../prompt_bank/data/t0_s0 --greedy ../prompt_bank/runs/Qwen3-1.7B__t0_s0__think \
      --out results/Qwen3-1.7B__t0_s0__think__sampling [--n-loop 100 --n-ctrl 32]
Resumes from out/responses.jsonl; writes out/summary.json and prints a table at the end.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import check_loops

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "prompt_bank"))
from pbank.answers import parse_choice, score            # noqa: E402
from pbank.local_model import LocalChatModel, log        # noqa: E402
from pbank.records import append_jsonl, load_prompts, read_jsonl   # noqa: E402

SAMPLING = {"do_sample": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0}


def h(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:12], 16)


def stratified(rows, regime, n, salt):
    """n rows spread over regimes in proportion to their counts, picked by hash (deterministic)."""
    by = defaultdict(list)
    for r in rows:
        by[regime[r["uid"]]].append(r)
    for g in by.values():
        g.sort(key=lambda r: h(salt + r["uid"]))
    total = len(rows)
    quota = {k: max(1, round(n * len(g) / total)) for k, g in by.items()}
    while sum(quota.values()) > n:                         # trim rounding excess from the largest group
        quota[max(quota, key=quota.get)] -= 1
    return [r for k, g in sorted(by.items()) for r in g[:quota[k]]]


def select(greedy_rows, regime, n_loop, n_ctrl):
    looping = [r for r in greedy_rows if r["hit_token_cap"] and check_loops.is_looping(r["generated_ids"])]
    answered = [r for r in greedy_rows if r["parse_status"] == "ok" and not r["hit_token_cap"]]
    pick = [(r, "loop") for r in stratified(looping, regime, n_loop, "loop")] + \
           [(r, "ctrl") for r in stratified(answered, regime, n_ctrl, "ctrl")]
    pick.sort(key=lambda p: h("order" + p[0]["uid"]))        # mix groups across batches
    return pick


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", required=True, type=Path)
    ap.add_argument("--greedy", required=True, type=Path, help="the greedy pbank run dir (responses.jsonl, run.json)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-loop", type=int, default=100)
    ap.add_argument("--n-ctrl", type=int, default=32)
    a = ap.parse_args()

    greedy_meta = json.loads((a.greedy / "run.json").read_text())["settings"]
    greedy_rows = read_jsonl(a.greedy / "responses.jsonl")
    greedy = {r["uid"]: r for r in greedy_rows}
    records = {r.uid: r for r in load_prompts(a.bank)}
    regime = {u: r.regime for u, r in records.items()}
    pick = select(greedy_rows, regime, a.n_loop, a.n_ctrl)
    group = {r["uid"]: g for r, g in pick}
    thinking = bool(greedy_meta["thinking"])
    cap, bs = greedy_meta["max_new_tokens"], greedy_meta["batch_size"]

    a.out.mkdir(parents=True, exist_ok=True)
    settings = {"greedy_run": str(a.greedy), "model": greedy_meta["model"], "thinking": thinking, "max_new_tokens": cap,
                "batch_size": bs, "sampling": SAMPLING, "n_loop": a.n_loop, "n_ctrl": a.n_ctrl,
                "uids": [r["uid"] for r, _ in pick]}
    sj = a.out / "settings.json"
    if sj.exists() and json.loads(sj.read_text()) != settings:
        sys.exit(f"{a.out} holds a run with other settings; use another --out")
    sj.write_text(json.dumps(settings, indent=2) + "\n")

    resp = a.out / "responses.jsonl"
    done = {r["uid"] for r in read_jsonl(resp)} if resp.exists() else set()
    todo = [r["uid"] for r, _ in pick if r["uid"] not in done]
    log(f"{len(pick)} selected ({a.n_loop} looping, {a.n_ctrl} answered controls), {len(done)} done, {len(todo)} to run")
    if todo:
        import torch
        model = LocalChatModel(greedy_meta["model"])
        orig = model.model.generate

        def sampled_generate(**kw):
            kw.update(SAMPLING)
            return orig(**kw)
        model.model.generate = sampled_generate
        for i in range(0, len(todo), bs):
            uids = todo[i:i + bs]
            torch.manual_seed(h("batch|" + "|".join(uids)) % 2**31)
            replies = model.generate([records[u].text for u in uids], thinking=thinking, max_new_tokens=cap)
            rows = []
            for u, rep in zip(uids, replies):
                parsed = parse_choice(rep.answer, records[u].labels)
                s = score(records[u], parsed)
                rows.append({"uid": u, "group": group[u], "regime": regime[u], "parse_status": parsed.status,
                             "chosen_option": s.chosen_option, "greedy_chosen_option": greedy[u]["chosen_option"],
                             "n_generated": rep.n_generated, "hit_token_cap": rep.hit_token_cap,
                             "looping": check_loops.is_looping(rep.generated_ids), "seconds": round(rep.seconds, 2),
                             "answer": rep.answer, "generated_ids": rep.generated_ids})
            append_jsonl(resp, rows)
            log(f"  [{len(done) + i + len(uids)}/{len(pick)}] {sum(r['parse_status'] == 'ok' for r in rows)}/{len(rows)} "
                f"parsed, max {max(r['n_generated'] for r in rows)} tokens, {replies[0].seconds:.1f} s/prompt")

    rows = read_jsonl(resp)
    summary = {}
    for g in ("loop", "ctrl"):
        rs = [r for r in rows if r["group"] == g]
        both = [r for r in rs if r["chosen_option"] and r["greedy_chosen_option"]]
        summary[g] = {"n": len(rs), "answered": sum(r["parse_status"] == "ok" for r in rs),
                      "hit_cap": sum(r["hit_token_cap"] for r in rs), "looping": sum(r["looping"] for r in rs),
                      "median_tokens": sorted(r["n_generated"] for r in rs)[len(rs) // 2] if rs else None,
                      "same_choice_as_greedy": f"{sum(r['chosen_option'] == r['greedy_chosen_option'] for r in both)}"
                                               f"/{len(both)}"}
        by_reg = defaultdict(lambda: [0, 0])
        for r in rs:
            by_reg[r["regime"]][0] += 1
            by_reg[r["regime"]][1] += r["parse_status"] == "ok"
        summary[g]["answered_by_regime"] = {k: f"{v[1]}/{v[0]}" for k, v in sorted(by_reg.items())}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
