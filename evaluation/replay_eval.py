"""Evaluation runner, replay readout: every method and null at +x and -x on the verification grid.

Uses only the extraction pipeline's documented interface (extraction_design/README.md; steering/README.md):
`steer.methods.load`, `Method.refit`, `steer.edits.LinearEdit`, `steer.hooks.replay_logits` / `steering`, and
`pbank.local_model.LocalChatModel`. The answer label is located here independently (regex on the answer text after the
end-of-thinking token), not with extraction's locator, so the two can be cross-checked.

Stages (each resumable; outputs in --out):
  items      the prompts that can be read out (parsed baseline reply) with their label tokens; unanswered ones listed
  calibrate  strength for push methods on a held-out fold of the extraction split (P4): CAA refitted without the
             fold, x swept on the fold's scored prompts, smallest x whose mean odd effect reaches --target-logit
  replay     every arm (methods, controls, nulls) at +x and -x, plus the unsteered pass: y = logit(far) - logit(near)
             at the answer slot, and the label mass p(far) + p(near)
  gencheck   steered generation on a subset (thinking as in the baseline): choices and health, for the replay-vs-
             generation agreement check

Usage: see run_local.sh. analyze.py turns the outputs into the report card.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

import nulls
import shared_draws as sd

CHOICE_RE = re.compile(r"I\s+choose\s*:?[\s*_`]*\(?([ab12])(?![a-z0-9])", re.I)
SCORED = ("near_only", "both")


# ------------------------------------------------------------------ io helpers

def read_jsonl(p: Path) -> list[dict]:
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_json(p: Path, obj) -> None:
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    tmp.replace(p)                                          # atomic: an interrupted write never looks finished


def write_jsonl_atomic(p: Path, rows) -> None:
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.replace(p)


def savez_atomic(p: Path, **arrays) -> None:
    tmp = p.with_name(p.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(p)


def find_key(obj, names):
    """First value under any of `names`, searched recursively (run.json layout is not documented field by field)."""
    if isinstance(obj, dict):
        for n in names:
            if n in obj:
                return obj[n]
        for v in obj.values():
            r = find_key(v, names)
            if r is not None:
                return r
    return None


def run_settings(run_dir: Path) -> dict:
    meta = json.loads((run_dir / "run.json").read_text())
    s = {"model_id": find_key(meta, ["model_id", "model"]), "thinking": find_key(meta, ["thinking"]),
         "max_new_tokens": find_key(meta, ["max_new_tokens"]), "batch_size": find_key(meta, ["batch_size"])}
    # Runs from before pbank's seeded sampling (PR 19) record no decoding: they were greedy.
    s["decoding"] = find_key(meta, ["decoding"]) or "greedy"
    if isinstance(s["model_id"], dict):
        s["model_id"] = find_key(s["model_id"], ["model_id", "id", "name"])
    missing = [k for k, v in s.items() if v is None]
    if missing:
        sys.exit(f"run.json: could not find {missing}")
    return s


def gen_decodings(decoding, n_samples: int) -> list:
    """The pbank Decoding of each generation-check sample, from the baseline run's recorded setting.

    Greedy (or a run that records none): one deterministic pass, whatever n_samples says. Sampling: n_samples
    passes with the run's temperature / top-p / top-k and seeds seed, seed+1, ... pbank re-seeds every batch from
    the seed and the batch's prompts, so every arm sees the same random numbers for the same sample and batch
    (paired comparisons across arms)."""
    from pbank.local_model import GREEDY, Decoding
    if decoding in (None, "greedy") or (isinstance(decoding, dict) and decoding.get("mode") == "greedy"):
        return [GREEDY]
    if not isinstance(decoding, dict) or decoding.get("mode") != "sample":
        raise ValueError(f"run.json: unknown decoding setting {decoding!r}")
    if n_samples < 1:
        raise ValueError("--gen-samples must be >= 1")
    return [Decoding(mode="sample", temperature=decoding["temperature"], top_p=decoding["top_p"],
                     top_k=decoding["top_k"], seed=decoding["seed"] + k, source=decoding.get("source"))
            for k in range(n_samples)]


def last_per_uid(rows: list[dict]) -> dict:
    out = {}
    for r in rows:
        out[r["uid"]] = r
    return out


# ------------------------------------------------------------------ model

class Model:
    def __init__(self, model_id: str):
        from pbank.local_model import LocalChatModel
        self.lcm = LocalChatModel(model_id)
        self.model = self.lcm.model
        self.tok = getattr(self.lcm, "tokenizer", None)
        if self.tok is None:
            from transformers import AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model_id = model_id

    def prompt_ids(self, text: str, thinking: bool) -> list[int]:
        return self.tok(self.lcm.chat_text(text, thinking), add_special_tokens=False)["input_ids"]

    def end_think_id(self) -> int | None:
        for t in ("</think>", "<channel|>"):
            i = self.tok.convert_tokens_to_ids(t)
            if isinstance(i, int) and i != self.tok.unk_token_id and i is not None:
                return i
        return None


# ------------------------------------------------------------------ items: where to read the answer

def locate_label(tok, gen_ids: list[int], end_think: int | None, thinking: bool, max_answer_tokens: int = 400):
    """(index of the label token in gen_ids, chosen char) or (None, reason). Independent of extraction's locator."""
    if thinking:
        if end_think is None or end_think not in gen_ids:
            return None, "thinking never closed"
        start = gen_ids.index(end_think) + 1
    else:
        start = gen_ids.index(end_think) + 1 if (end_think is not None and end_think in gen_ids) else 0
    ans = gen_ids[start:start + max_answer_tokens]
    ends, text = [], ""
    for i in range(len(ans)):
        text = tok.decode(ans[: i + 1], skip_special_tokens=False)
        ends.append(len(text))
    m = CHOICE_RE.search(text)
    if not m:
        return None, "no 'I choose:' in the answer"
    c = m.start(1)
    i = next(k for k, e in enumerate(ends) if e > c)
    return start + i, m.group(1).lower()


def label_pair(tok, label_id: int, chosen: str, other: str) -> int:
    """Token id of the other label in the same style: the generated label token with its character swapped."""
    s = tok.decode([label_id])
    if chosen not in s.lower():
        raise ValueError(f"label token {s!r} does not contain {chosen!r}")
    i = s.lower().index(chosen)
    s2 = s[:i] + (other.upper() if s[i].isupper() else other) + s[i + 1:]
    ids = tok.encode(s2, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"other label {s2!r} is {len(ids)} tokens")
    return ids[0]


def build_items(args, M: Model, settings: dict) -> None:
    out = Path(args.out)
    if (out / "items.jsonl").exists():
        print("items: exist, skipping")
        return
    bank = {r["uid"]: r for r in read_jsonl(Path(args.bank) / "prompts.jsonl")}
    resp = last_per_uid(read_jsonl(Path(args.run) / "responses.jsonl"))
    end_think = M.end_think_id()
    items, skipped = [], []
    for uid, r in sorted(resp.items()):
        b = bank[uid]
        if b["regime"] == "no_horizon":
            continue
        gen = r.get("generated_ids") or []
        if r.get("parse_status") != "ok":
            skipped.append({"uid": uid, "split": b["split"], "reason": f"pbank parse_status {r.get('parse_status')}",
                            "hit_token_cap": r.get("hit_token_cap"), "n_generated": r.get("n_generated")})
            continue
        idx, what = locate_label(M.tok, gen, end_think, settings["thinking"])
        if idx is None:
            skipped.append({"uid": uid, "split": b["split"], "reason": what, "parse_status": r.get("parse_status"),
                            "hit_token_cap": r.get("hit_token_cap"), "n_generated": r.get("n_generated")})
            continue
        chars = [lab[0].lower() for lab in b["labels"]]
        far_char = chars[1] if b["near_first"] else chars[0]
        near_char = chars[0] if b["near_first"] else chars[1]
        if what not in chars:
            skipped.append({"uid": uid, "split": b["split"], "reason": f"chose {what!r}, labels {chars}"})
            continue
        other = near_char if what == far_char else far_char
        try:
            other_id = label_pair(M.tok, gen[idx], what, other)
        except ValueError as e:
            skipped.append({"uid": uid, "split": b["split"], "reason": str(e)})
            continue
        far_id = gen[idx] if what == far_char else other_id
        near_id = other_id if what == far_char else gen[idx]
        theirs = (r.get("choice_label") or "")[:1].lower()
        items.append({
            "uid": uid, "split": b["split"], "regime": b["regime"], "pair_id": b.get("pair_id"),
            "near_first": b["near_first"], "label_style": "letters" if chars[0] == "a" else "digits",
            "h_over_far_delay": b["h_over_far_delay"], "horizon_grid_value": b["horizon_grid_value"],
            "reward_multiplier": b["reward_multiplier"], "correct_option": b.get("correct_option"),
            "label_index": idx, "chosen": "far" if what == far_char else "near",
            "far_id": far_id, "near_id": near_id, "label_token": M.tok.decode([gen[idx]]),
            "agrees_with_pbank_parse": theirs == what if theirs else None,
            "parse_status": r.get("parse_status"), "n_generated": r.get("n_generated"),
        })
    write_jsonl_atomic(out / "skipped.jsonl", skipped)
    write_jsonl_atomic(out / "items.jsonl", items)          # last: its presence marks the stage as done
    disagree = sum(1 for it in items if it["agrees_with_pbank_parse"] is False)
    print(f"items: {len(items)} readable, {len(skipped)} skipped, {disagree} disagree with pbank's parse")
    if args.extraction_index:
        theirs = {r["uid"]: r for r in read_jsonl(Path(args.extraction_index))}
        both = [it for it in items if it["uid"] in theirs]
        same = [it for it in both if theirs[it["uid"]].get("label_index") == it["label_index"]]
        mine_only = sorted({it["uid"] for it in items} - set(theirs))
        theirs_only = sorted(set(theirs) - {it["uid"] for it in items}
                             - {u for u in theirs if u.endswith("-null")})
        check = {"n_both": len(both), "same_label_index": len(same),
                 "differ": [{"uid": it["uid"], "ours": it["label_index"], "theirs": theirs[it["uid"]].get("label_index")}
                            for it in both if it not in same][:50],
                 "readable_here_not_extracted": mine_only[:50], "extracted_not_readable_here": theirs_only[:50]}
        write_json(out / "label_crosscheck.json", check)
        print(f"label cross-check vs extraction: {len(same)}/{len(both)} same index; "
              f"{len(mine_only)} only here, {len(theirs_only)} only there")


def load_items(out: Path, split: str) -> list[dict]:
    return [it for it in read_jsonl(out / "items.jsonl") if it["split"] == split]


# ------------------------------------------------------------------ readout

class Replayer:
    """y = logit(far) - logit(near) and label mass at the answer slot, per item, under one edit and request."""

    def __init__(self, M: Model, run_dir: Path, bank_dir: Path, thinking: bool, layer: int):
        from steer.hooks import replay_logits
        self.replay_logits = replay_logits
        self.M, self.layer = M, layer
        bank = {r["uid"]: r for r in read_jsonl(bank_dir / "prompts.jsonl")}
        self.resp = last_per_uid(read_jsonl(run_dir / "responses.jsonl"))
        self.bank, self.thinking = bank, thinking
        self._cache = {}

    def seqs(self, it):
        if it["uid"] not in self._cache:
            p = self.M.prompt_ids(self.bank[it["uid"]]["text"], self.thinking)
            a = self.resp[it["uid"]]["generated_ids"][: it["label_index"]]
            self._cache[it["uid"]] = (p, a)
        return self._cache[it["uid"]]

    @torch.no_grad()
    def run(self, items, edit, x, positions="all"):
        y = np.empty(len(items), np.float32)
        mass = np.empty(len(items), np.float32)
        for n, it in enumerate(items):
            p, a = self.seqs(it)
            lg = self.replay_logits(self.M.model, p, a, edit, x, self.layer, positions=positions)
            lg = lg.float()
            y[n] = float(lg[it["far_id"]] - lg[it["near_id"]])
            pr = torch.softmax(lg, -1)
            mass[n] = float(pr[it["far_id"]] + pr[it["near_id"]])
        return y, mass

    @torch.no_grad()
    def plain(self, it):
        """Logits with no hook at all (for the zero-edit check)."""
        p, a = self.seqs(it)
        ids = torch.tensor([p + a], device=self.M.model.device)
        return self.M.model(input_ids=ids).logits[0, -1].float()


# ------------------------------------------------------------------ arms

def method_dirs(methods_dir: Path) -> dict:
    out = {}
    for d in sorted(methods_dir.iterdir()):
        if (d / "method.json").exists():
            out[json.loads((d / "method.json").read_text())["method"]] = d
    return out


def shuffled_horizons(rows: list[dict], bank: dict, seed: int) -> dict:
    """Grid values permuted within strata of (option order, regime)."""
    rng = np.random.default_rng(seed)
    strata = {}
    for r in rows:
        b = bank[r["uid"]]
        strata.setdefault((b["near_first"], b["regime"]), []).append(r["uid"])
    out = {}
    for uids in strata.values():
        vals = [bank[u]["horizon_grid_value"] for u in uids]
        for u, v in zip(uids, rng.permutation(len(vals))):
            out[u] = vals[v]
    return out


def build_arms(args, cfg: dict) -> list[dict]:
    """Arm specs (no tensors): what to load and at which request. Edits are built lazily in make_edit."""
    md = method_dirs(Path(args.methods))
    arms = []
    for name in ("caa", "reward", "choice"):
        if name in md:
            arms.append({"arm": name, "family": "method" if name == "caa" else "control", "kind": "push",
                         "x": cfg["x_push"], "src": str(md[name])})
    for name in ("curve-displace", "curve-snap"):
        if name in md:
            arms.append({"arm": name, "family": "method", "kind": "target", "x": cfg["x_shift"], "src": str(md[name])})
    if "curve-snap" in md:
        arms.append({"arm": "curve-snap@0", "family": "control", "kind": "target", "x": 0.0, "src": str(md["curve-snap"]),
                     "single": True})
    K = args.k
    for fam in ("isotropic", "covariance", "real_diff"):
        for i in range(K):
            arms.append({"arm": f"{fam}[{i}]", "family": fam, "kind": "push", "x": cfg["x_push"], "row": i})
    if "curve-displace" in md:
        for i in range(K):
            arms.append({"arm": f"in_span[{i}]", "family": "in_span", "kind": "push", "x": cfg["x_push"], "row": i})
    for base in ("caa", "curve-displace", "curve-snap"):
        if base in md:
            for i in range(K):
                arms.append({"arm": f"shuffled:{base}[{i}]", "family": f"shuffled:{base}",
                             "kind": "push" if base == "caa" else "target",
                             "x": cfg["x_push"] if base == "caa" else cfg["x_shift"], "src": str(md[base]), "seed": i})
    return arms


class EditFactory:
    def __init__(self, args):
        from steer.edits import LinearEdit
        from steer.methods import load
        self.LinearEdit, self.load = LinearEdit, load
        self.args = args
        self.bank = {r["uid"]: r for r in read_jsonl(Path(args.bank) / "prompts.jsonl")}
        md = method_dirs(Path(args.methods))
        self.caa = load(md["caa"])
        self.typical_norm = float(self.caa.meta["request"]["typical_norm"])
        self.positions = self.caa.positions
        dpath = Path(args.draws)
        self.draws = np.load(dpath / "draws.npz")
        self.span = None
        if "curve-displace" in md:
            arr = np.load(md["curve-displace"] / "arrays.npz")
            U = arr["U"]
            U = U if U.shape[0] < U.shape[1] else U.T
            self.span = np.linalg.qr(U.T)[0].T             # orthonormal rows spanning the curve's subspace
        self._methods = {}

    def method(self, src):
        if src not in self._methods:
            self._methods[src] = self.load(Path(src))
        return self._methods[src]

    def make(self, arm: dict):
        fam = arm["family"]
        if fam in ("method", "control"):
            m = self.method(arm["src"])
            return m.edit, m.positions
        if fam in ("isotropic", "covariance", "real_diff"):
            v = self.draws[fam][arm["row"]].astype(np.float64)
            return self.LinearEdit(v, self.typical_norm), self.positions
        if fam == "in_span":
            key = sd.seed_key(self.caa.meta.get("fitted_on", {}).get("model", {}).get("model_id", "model"),
                              self.caa.layer, self.caa.site, "in_span")
            V = sd.normals(sd.stream(key), (self.args.k, self.span.shape[0])) @ self.span
            v = V[arm["row"]] / np.linalg.norm(V[arm["row"]])
            return self.LinearEdit(v, self.typical_norm), self.positions
        if fam.startswith("shuffled:"):
            m = self.method(arm["src"])
            hz = shuffled_horizons(m.fit_uids(), self.bank, seed=1000 + arm["seed"])
            r = m.refit(Path(self.args.activations), Path(self.args.bank), horizons=hz)
            return r.edit, r.positions
        raise ValueError(fam)


# ------------------------------------------------------------------ stages

def stage_calibrate(args, M, settings, rep: Replayer) -> dict:
    out = Path(args.out)
    if (out / "calibration.json").exists():
        return json.loads((out / "calibration.json").read_text())
    from steer.methods import load
    caa = load(method_dirs(Path(args.methods))["caa"])
    rows = caa.fit_uids()
    fold = args.calib_fold
    train = [r["uid"] for r in rows if r["fold"] != fold]
    held = {r["uid"] for r in rows if r["fold"] == fold}
    items = [it for it in load_items(out, "extraction") if it["uid"] in held and it["regime"] in SCORED]
    caa_f = caa.refit(Path(args.activations), Path(args.bank), uids=train)
    _, m0 = rep.run(items, caa_f.edit, 0.0, caa_f.positions)
    mass0 = float(m0.mean())
    sweep = []
    chosen = None
    healthy = []
    for x in args.x_grid:
        yp, mp = rep.run(items, caa_f.edit, x, caa_f.positions)
        ym, mm = rep.run(items, caa_f.edit, -x, caa_f.positions)
        o = (yp - ym) / 2
        nf = np.array([it["near_first"] for it in items])
        # average the two option orders' means, so a push toward one label (letter, C1) cancels out
        odd = float((o[nf].mean() + o[~nf].mean()) / 2) if nf.any() and (~nf).any() else float(o.mean())
        mass = float(min(mp.mean(), mm.mean()))
        ok = mass >= args.min_mass_ratio * mass0
        sweep.append({"x": x, "mean_odd_logit": odd, "min_label_mass": mass, "healthy": ok})
        print(f"calibrate: x={x:g} mean odd {odd:+.3f} logits, label mass {mass:.3f} (unsteered {mass0:.3f})"
              f"{'' if ok else ' UNHEALTHY'} on {len(items)} held-out prompts")
        if ok:
            healthy.append(x)
            if chosen is None and abs(odd) >= args.target_logit:
                chosen = x
    # Size, not sign, picks the strength: the sign is a result (a method may steer backwards), reported by analyze.py.
    # Never beyond the health bound: if the target isn't reached while healthy, take the largest healthy strength.
    rule = "smallest healthy x with |mean odd| >= target"
    if chosen is None:
        chosen = healthy[-1] if healthy else args.x_grid[0]
        rule = "target not reached while healthy: largest healthy x" if healthy else "no healthy x: smallest x"
    cal = {"fold": fold, "n_train": len(train), "n_heldout_scored": len(items), "target_logit": args.target_logit,
           "min_mass_ratio": args.min_mass_ratio, "unsteered_label_mass": mass0, "sweep": sweep, "x_push": chosen,
           "rule": rule, "reached_target": rule.startswith("smallest healthy"), "x_shift": args.x_shift}
    write_json(out / "calibration.json", cal)
    return cal


def stage_replay(args, M, settings, rep: Replayer, cal: dict) -> None:
    out = Path(args.out)
    adir = out / "arms"
    adir.mkdir(exist_ok=True)
    items = load_items(out, "verification")
    fac = EditFactory(args)
    # unsteered pass and the zero-edit check against a hook-free forward pass (N0)
    if not (adir / "_none.npz").exists():
        y0, m0 = rep.run(items, fac.caa.edit, 0.0, fac.positions)
        n0 = min(args.n0_check, len(items))
        diffs = [float((rep.run([it], fac.caa.edit, 0.0, fac.positions)[0][0])
                       - float(rep.plain(it)[it["far_id"]] - rep.plain(it)[it["near_id"]])) for it in items[:n0]]
        agrees = float(np.mean([(y > 0) == (it["chosen"] == "far") for y, it in zip(y0, items)]))
        savez_atomic(adir / "_none.npz", y=y0, mass=m0)
        write_json(out / "n0_check.json", {"n_checked": n0, "max_abs_diff_vs_hook_free": max(map(abs, diffs)) if diffs else None,
                                          "replay_matches_generated_choice": agrees})
        print(f"N0: zero edit vs hook-free max |diff| {max(map(abs, diffs)) if diffs else float('nan'):.3g}; "
              f"unsteered replay picks the generated label on {agrees:.1%} of items")
    arms = build_arms(args, cal)
    with open(out / "arms.jsonl", "w") as f:
        for a in arms:
            f.write(json.dumps(a) + "\n")
    t0 = time.time()
    for n, arm in enumerate(arms):
        path = adir / (hashlib.sha1(arm["arm"].encode()).hexdigest()[:12] + ".npz")
        if path.exists():
            continue
        edit, positions = fac.make(arm)
        rec = {}
        rec["y_plus"], rec["mass_plus"] = rep.run(items, edit, arm["x"], positions)
        if not arm.get("single"):
            rec["y_minus"], rec["mass_minus"] = rep.run(items, edit, -arm["x"], positions)
        savez_atomic(path, **rec)
        el = time.time() - t0
        print(f"[{n + 1}/{len(arms)}] {arm['arm']} x={arm['x']:g}  ({el / 60:.1f} min)", flush=True)


def stage_gencheck(args, M, settings, cal: dict) -> None:
    """Steered generation on a subset of verification pairs: none, caa +-x, isotropic[0] +-x, curve-displace +-shift."""
    from steer.hooks import steering
    out = Path(args.out) / "gencheck"
    out.mkdir(exist_ok=True)
    items = load_items(Path(args.out), "verification")
    pairs = {}
    for it in items:
        pairs.setdefault(it["pair_id"], []).append(it)
    full = [p for p in sorted(pairs) if len(pairs[p]) == 2]
    pick = [p for i, p in enumerate(full) if i % max(1, len(full) // args.gen_pairs) == 0][: args.gen_pairs]
    sub = [it for p in pick for it in pairs[p]]
    bank = {r["uid"]: r for r in read_jsonl(Path(args.bank) / "prompts.jsonl")}
    fac = EditFactory(args)
    md = method_dirs(Path(args.methods))
    arms = [("none", None, 0.0)]
    for nm, spec in (("caa", {"family": "method", "src": str(md["caa"])}),
                     ("isotropic[0]", {"family": "isotropic", "row": 0}),
                     ("curve-displace", {"family": "method", "src": str(md.get("curve-displace", ""))})):
        if spec.get("src") == "":
            continue
        x = cal["x_shift"] if nm.startswith("curve") else cal["x_push"]
        arms += [(f"{nm}+", spec, x), (f"{nm}-", spec, -x)]
    end_think = M.end_think_id()
    decodings = gen_decodings(settings["decoding"], args.gen_samples)
    if len(decodings) < args.gen_samples:
        print(f"gencheck: the baseline run is greedy, so one pass (not {args.gen_samples}): extra samples would be identical")
    for name, spec, x in arms:
        path = out / f"{name}.jsonl"
        if path.exists():
            continue
        rows = []
        t0 = time.time()
        for k, dec in enumerate(decodings):
            for s in range(0, len(sub), args.gen_batch):
                batch = sub[s: s + args.gen_batch]
                texts = [bank[it["uid"]]["text"] for it in batch]
                if spec is None:
                    reps = M.lcm.generate(texts, thinking=settings["thinking"], max_new_tokens=settings["max_new_tokens"],
                                          decoding=dec)
                else:
                    edit, positions = fac.make(spec)
                    with steering(M.model, edit, x, fac.caa.layer, positions=positions):
                        reps = M.lcm.generate(texts, thinking=settings["thinking"],
                                              max_new_tokens=settings["max_new_tokens"], decoding=dec)
                for it, r in zip(batch, reps):
                    ids = list(getattr(r, "generated_ids", []) or [])
                    idx, what = locate_label(M.tok, ids, end_think, settings["thinking"]) if ids else (None, "no ids")
                    b = bank[it["uid"]]
                    chars = [lab[0].lower() for lab in b["labels"]]
                    far_char = chars[1] if b["near_first"] else chars[0]
                    rows.append({"uid": it["uid"], "arm": name, "x": x, "sample": k, "seed": dec.seed,
                                 "parsed": idx is not None,
                                 "chose_far": (what == far_char) if idx is not None else None,
                                 "reason": None if idx is not None else what,
                                 "n_generated": len(ids), "hit_cap": len(ids) >= settings["max_new_tokens"]})
        write_jsonl_atomic(path, rows)
        ok = [r for r in rows if r["parsed"]]
        print(f"gencheck {name}: {len(ok)}/{len(rows)} parsed, P(far) "
              f"{np.mean([r['chose_far'] for r in ok]) if ok else float('nan'):.2f}, {(time.time() - t0) / 60:.1f} min")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("stages", nargs="+", choices=["items", "calibrate", "replay", "gencheck"])
    p.add_argument("--bank", required=True), p.add_argument("--run", required=True)
    p.add_argument("--activations", required=True), p.add_argument("--methods", required=True)
    p.add_argument("--draws", required=True), p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, default=sd.K_GEN, help="draws per null family (rows [:k] of the shared draws)")
    p.add_argument("--x-grid", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.2, 0.4, 0.8])
    p.add_argument("--target-logit", type=float, default=1.0)
    p.add_argument("--min-mass-ratio", type=float, default=0.95,
                   help="health bound: label mass at +-x must stay >= this fraction of the unsteered mass")
    p.add_argument("--extraction-index", help="extract.py's index.jsonl, to cross-check our label locator")
    p.add_argument("--x-shift", type=float, default=1.0, help="request for target methods, decades")
    p.add_argument("--calib-fold", type=int, default=0)
    p.add_argument("--n0-check", type=int, default=8)
    p.add_argument("--gen-pairs", type=int, default=16), p.add_argument("--gen-batch", type=int, default=8)
    p.add_argument("--gen-samples", type=int, default=1,
                   help="generation check on a sampled baseline: passes per arm, seeds seed..seed+n-1 (greedy: 1)")
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    settings = run_settings(Path(args.run))
    write_json(out / "settings.json", {"args": vars(args), "run": settings})
    M = Model(settings["model_id"])
    from steer.methods import load
    layer = load(method_dirs(Path(args.methods))["caa"]).layer
    rep = Replayer(M, Path(args.run), Path(args.bank), settings["thinking"], layer)
    if "items" in args.stages:
        build_items(args, M, settings)
    cal = None
    if "calibrate" in args.stages or "replay" in args.stages or "gencheck" in args.stages:
        cal = stage_calibrate(args, M, settings, rep)
    if "replay" in args.stages:
        stage_replay(args, M, settings, rep, cal)
    if "gencheck" in args.stages:
        stage_gencheck(args, M, settings, cal)


if __name__ == "__main__":
    main()
