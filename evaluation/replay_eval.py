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
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import numpy as np
import torch

import nulls
import shared_draws as sd

CHOICE_RE = re.compile(r"I\s+choose\s*:?[\s*_`]*\(?([ab12])(?![a-z0-9])", re.I)
SCORED = ("near_only", "both")
COMPARISON_PUSH = ("caa-none", "caa-near_only", "caa-both", "caa-within")   # within-regime CAA (PR 18)

# N6 (damage-matched null): "damage" is the mean next-token KL(unedited || edited) over every token of these texts,
# edited at the method's own positions. Plain prose with no amounts, options, durations or planning, so that the KL
# measures disruption of the model in general, not a change in what it thinks of the task.
NEUTRAL_TEXTS = (
    "The library on the corner has tall windows and wooden shelves. In the reading room, people sit at long tables "
    "with books and notebooks, and the only sound is the turning of pages.",
    "Bread is made from flour, water, salt and yeast. The dough is kneaded until it is smooth, left to rise, shaped "
    "into a loaf and baked until the crust is golden brown.",
    "A bicycle chain carries the force of the pedals to the rear wheel. Keeping it clean and lightly oiled makes the "
    "ride quieter and helps the gears change smoothly.",
    "Leaves are green because they contain chlorophyll, a pigment that absorbs red and blue light. Plants use that "
    "light to turn water and carbon dioxide into sugar and oxygen.",
    "The river runs through the middle of the town. Two stone bridges cross it, and on summer evenings families walk "
    "along the bank to watch the ducks and the small boats.",
    "In chess, each player starts with sixteen pieces. The knight moves in an L shape, the bishop moves diagonally, "
    "and the rook moves in straight lines along ranks and files.",
    "The museum's new exhibit shows pottery from many regions. Some bowls are painted with geometric patterns, while "
    "others are left plain to show the colour and texture of the clay.",
    "A good map shows roads, rivers and hills with clear symbols. The legend explains each symbol, and the scale bar "
    "lets the reader estimate the distance between two places.",
)
N6_TOLERANCE = 0.10        # a draw is matched when its KL is within 10% of the method's

# The prompting baseline (evaluation_design §9 stage 5, "vs prompting baseline"): one sentence appended as the last line
# of the last user message. Wording from Jared's E22 (jared/EvalsPromptsV2.md §10-11 on branch jared-eval-types), where
# on Qwen3-8B the disposition pair moved the choice as far as the best steering but also moved rows it should not, and
# the shift pair was mostly horizon-specific.
PROMPTS = {
    "neutral": "Think carefully about this decision.",
    "long": "When you decide, be more long-term-minded than you otherwise would be.",
    "short": "When you decide, be more short-term-minded than you otherwise would be.",
    "shift_long": "Act as if the time horizon were 100 times longer than stated.",
    "shift_short": "Act as if the time horizon were 100 times shorter than stated.",
}
# arm -> (sentence at +x, sentence at -x or None); + is the longer horizon, as for the steering methods
PROMPT_ARMS = {"prompt:disposition": ("long", "short"), "prompt:shift": ("shift_long", "shift_short"),
               "prompt:neutral": ("neutral", None)}


def with_sentence(prompt, sentence: str):
    """A user text, or a conversation whose last message is the user's, with `sentence` as its new last line."""
    if isinstance(prompt, str):
        return f"{prompt}\n{sentence}"
    if prompt[-1]["role"] != "user":
        raise ValueError("the last message must be the user's")
    return [*prompt[:-1], {**prompt[-1], "content": f"{prompt[-1]['content']}\n{sentence}"}]


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


@contextmanager
def weights_in(model, dtype: str):
    """Run the block with the model's weights in `dtype`, then put them back. The replay readout differences are
    small (CAA at the calibrated x moves y by ~0.02 logit per item), and bf16 rounding alone moves y by ~0.3 logit per
    item vs float32 (Qwen3-1.7B, 2026-11-23), so the readout runs in float32. bf16 -> float32 -> bf16 is exact, so
    generation afterwards uses the very same weights."""
    before = model.dtype
    model.to(getattr(torch, dtype))
    try:
        yield
    finally:
        model.to(before)


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

    def __init__(self, M: Model, run_dir: Path, bank_dir: Path, thinking: bool, layer: int, batch_size: int = 16):
        from steer.hooks import steering
        self.steering = steering
        self.M, self.layer, self.batch_size = M, layer, batch_size
        bank = {r["uid"]: r for r in read_jsonl(bank_dir / "prompts.jsonl")}
        self.resp = last_per_uid(read_jsonl(run_dir / "responses.jsonl"))
        self.bank, self.thinking = bank, thinking
        self._cache = {}

    def prompt_of(self, it):
        """What the model was given for this item (a subclass may return a conversation)."""
        return self.bank[it["uid"]]["text"]

    def seqs(self, it):
        if it["uid"] not in self._cache:
            p = self.M.prompt_ids(self.prompt_of(it), self.thinking)
            a = self.resp[it["uid"]]["generated_ids"][: it["label_index"]]
            self._cache[it["uid"]] = (p, a)
        return self._cache[it["uid"]]

    def _last_logits(self, seqs: list[list[int]]) -> torch.Tensor:
        """Next-token logits (float32 [n, vocab]) after each token sequence, batch_size sequences per forward pass.

        Right padding: under the causal mask a row's tokens never attend to the padding after them, and positions stay
        0..len-1, so each row reads as it would alone, up to batched bf16 arithmetic (stage_replay measures that gap).
        The steering hook leaves padding unedited (attention mask 0)."""
        device = self.M.model.device
        # rows sorted by length, so a batch pads little (thinking replies run from 100 to 4,000 tokens); the
        # readouts go back to the callers' order at the end
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
        out = []
        for s in range(0, len(order), self.batch_size):
            chunk = [seqs[i] for i in order[s: s + self.batch_size]]
            ids = torch.zeros(len(chunk), max(map(len, chunk)), dtype=torch.long)
            mask = torch.zeros_like(ids)
            for i, q in enumerate(chunk):
                ids[i, :len(q)] = torch.tensor(q)
                mask[i, :len(q)] = 1
            last = (mask.sum(1) - 1).to(device)
            # logits only at the rows' last positions: [rows, rows, vocab], row i's own readout on the diagonal
            lg = self.M.model(input_ids=ids.to(device), attention_mask=mask.to(device), use_cache=False,
                              logits_to_keep=last).logits
            rows = torch.arange(len(chunk), device=lg.device)
            out.append(lg[rows, rows].float())
        sorted_logits = torch.cat(out)
        back = torch.empty(len(order), dtype=torch.long)
        back[torch.tensor(order, dtype=torch.long)] = torch.arange(len(order))
        return sorted_logits[back.to(sorted_logits.device)]

    @staticmethod
    def _readout(items, lg: torch.Tensor):
        """y = logit(far) - logit(near) and label mass p(far) + p(near), per item, from [n, vocab] logits."""
        rows = torch.arange(len(items), device=lg.device)
        far = torch.tensor([it["far_id"] for it in items], device=lg.device)
        near = torch.tensor([it["near_id"] for it in items], device=lg.device)
        pr = torch.softmax(lg, -1)
        y = (lg[rows, far] - lg[rows, near]).cpu().numpy().astype(np.float32)
        mass = (pr[rows, far] + pr[rows, near]).cpu().numpy().astype(np.float32)
        return y, mass

    @torch.no_grad()
    def run_prompted(self, items, sentence: str):
        """y and label mass with `sentence` appended to the prompt and the generated reasoning kept as it was, so the
        prompt is read like an edit is: with the reasoning held fixed (no hook)."""
        seqs = [self.M.prompt_ids(with_sentence(self.prompt_of(it), sentence), self.thinking) + self.seqs(it)[1]
                for it in items]
        return self._readout(items, self._last_logits(seqs))

    @torch.no_grad()
    def run(self, items, edit, x, positions="all"):
        if positions != "all":
            # a batch has one pass but many prompt lengths, so the hook could not tell each row's prompt from its reply
            raise ValueError(f"batched replay supports positions='all' only, not {positions!r}")
        seqs = [p + a for p, a in map(self.seqs, items)]
        with self.steering(self.M.model, edit, x, self.layer, positions):
            lg = self._last_logits(seqs)
        return self._readout(items, lg)

    @torch.no_grad()
    def plain(self, it):
        """Logits with no hook at all (for the zero-edit check)."""
        p, a = self.seqs(it)
        ids = torch.tensor([p + a], device=self.M.model.device)
        return self.M.model(input_ids=ids).logits[0, -1].float()


# ------------------------------------------------------------------ arms

def arm_selected(arm: dict, wanted) -> bool:
    """Whether a list of wanted arms (--arms, a check set's `arms`) names this arm, its family or its method (the
    name before "[" or "@"); None selects every arm."""
    if wanted is None:
        return True
    names = {arm["arm"], arm["family"], arm["arm"].split("[")[0].split("@")[0]}
    return bool(names & set(wanted))


def method_dirs(methods_dir: Path, site: str | None = None) -> dict:
    """method name -> bundle directory, for the bundles in `methods_dir` (at `site` only, if given).

    A methods directory can hold the same method at several sites (the Gemma run bundles each method at the label
    and the prelabel token). Two bundles of one method are refused rather than one silently replacing the other."""
    out, sites = {}, {}
    for d in sorted(methods_dir.iterdir()):
        if not (d / "method.json").exists():
            continue
        meta = json.loads((d / "method.json").read_text())
        if site is not None and meta.get("site") != site:
            continue
        name = meta["method"]
        if name in out:
            sys.exit(f"{methods_dir} has {name} twice ({out[name].name} and {d.name}, sites {sites[name]} and "
                     f"{meta.get('site')}): pass --site to choose one")
        out[name], sites[name] = d, meta.get("site")
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
    md = method_dirs(Path(args.methods), getattr(args, "site", None))   # no site given: no site filter
    arms = []
    for name in ("caa", "reward", "choice"):
        if name in md:
            arms.append({"arm": name, "family": "method" if name == "caa" else "control", "kind": "push",
                         "x": cfg["x_push"], "src": str(md[name])})
    for name in COMPARISON_PUSH:
        if name in md:
            arms.append({"arm": name, "family": "comparison", "kind": "push", "x": cfg["x_push"], "src": str(md[name])})
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
    if getattr(args, "prompting", False):
        for name, (plus, minus) in PROMPT_ARMS.items():
            arms.append({"arm": name, "family": "prompting", "kind": "prompt", "x": 1.0, "plus": plus, "minus": minus,
                         **({"single": True} if minus is None else {})})
    n6_path = Path(args.out) / "n6.json"
    if n6_path.exists():                     # N6: isotropic draws at the strength that matches the method's damage
        for m, rec in json.loads(n6_path.read_text())["methods"].items():
            for d in rec["draws"]:
                if d["matched"]:
                    arm = {"arm": f"n6:{m}[{d['row']}]", "family": f"n6:{m}", "kind": "push", "x": d["x"], "row": d["row"]}
                    if d["x"] == cfg["x_push"] and d["row"] < K:
                        arm["same_as"] = f"isotropic[{d['row']}]"     # same vector, same strength: reuse, don't replay
                    arms.append(arm)
    return arms


class EditFactory:
    def __init__(self, args):
        from steer.edits import LinearEdit
        from steer.methods import load
        self.LinearEdit, self.load = LinearEdit, load
        self.args = args
        self.bank = {r["uid"]: r for r in read_jsonl(Path(args.bank) / "prompts.jsonl")}
        md = method_dirs(Path(args.methods), getattr(args, "site", None))
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
        if fam in ("method", "control", "comparison"):
            m = self.method(arm["src"])
            return m.edit, m.positions
        if fam in ("isotropic", "covariance", "real_diff") or fam.startswith("n6:"):
            v = self.draws["isotropic" if fam.startswith("n6:") else fam][arm["row"]].astype(np.float64)
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

def label_mass_health(mass_plus: float, mass_minus: float, mass0: float, ratio: float) -> tuple[float, bool]:
    """Health on the even part of the label mass (P2): (mass(+x) + mass(-x)) / 2 must stay >= ratio * unsteered.
    A mass change that reverses with the sign is steering, not damage: on Qwen3-1.7B, CAA at +x lowered the mass
    as much as -x raised it, and a rule on the smaller of the two stopped calibration at x = 0.02."""
    even = (mass_plus + mass_minus) / 2
    return even, even >= ratio * mass0


def stage_calibrate(args, M, settings, rep: Replayer) -> dict:
    out = Path(args.out)
    if (out / "calibration.json").exists():
        return json.loads((out / "calibration.json").read_text())
    from steer.methods import load
    caa = load(method_dirs(Path(args.methods), args.site)["caa"])
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
        mass, ok = label_mass_health(float(mp.mean()), float(mm.mean()), mass0, args.min_mass_ratio)
        sweep.append({"x": x, "mean_odd_logit": odd, "even_label_mass": mass, "healthy": ok,
                      "label_mass_plus": float(mp.mean()), "label_mass_minus": float(mm.mean())})
        print(f"calibrate: x={x:g} mean odd {odd:+.3f} logits, even label mass {mass:.3f} (unsteered {mass0:.3f})"
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


class Damage:
    """Next-token KL(unedited || edited), averaged over every token of NEUTRAL_TEXTS (N6's damage measure)."""

    def __init__(self, M: Model, layer: int):
        from steer.hooks import steering
        self.steering, self.M, self.layer = steering, M, layer
        dev = M.model.get_input_embeddings().weight.device
        self.ids = [torch.tensor([M.tok(t)["input_ids"]], device=dev) for t in NEUTRAL_TEXTS]
        with torch.no_grad():
            self.ref = [torch.log_softmax(M.model(input_ids=x).logits[0].float(), -1) for x in self.ids]

    @torch.no_grad()
    def kl(self, edit, x: float, positions="all") -> float:
        tot, n = 0.0, 0
        with self.steering(self.M.model, edit, x, self.layer, positions=positions):
            for x_ids, ref in zip(self.ids, self.ref):
                lq = torch.log_softmax(self.M.model(input_ids=x_ids).logits[0].float(), -1)
                tot += float((ref.exp() * (ref - lq)).sum(-1).sum())
                n += ref.shape[0]
        return tot / n

    def pm(self, edit, x: float, positions="all") -> float:
        """Mean of the KL at +x and -x: the damage of a ± pair, as the replay arms use it."""
        return (self.kl(edit, x, positions) + self.kl(edit, -x, positions)) / 2


def match_strength(f, target: float, x0: float, tol: float = N6_TOLERANCE, max_steps: int = 30):
    """A strength x > 0 with f(x) within `tol` (relative) of `target`, for f increasing in x: bracket by doubling or
    halving from x0, then bisect on log x. Returns (x, f(x), matched, steps). Every step costs one evaluation of f."""
    x, steps = x0, 0
    fx = f(x); steps += 1
    lo = hi = None
    while steps < max_steps:
        if abs(fx / target - 1) <= tol:
            return x, fx, True, steps
        if fx < target:
            lo = (x, fx)
            x = x * 2 if hi is None else float(np.sqrt(lo[0] * hi[0]))
        else:
            hi = (x, fx)
            x = x / 2 if lo is None else float(np.sqrt(lo[0] * hi[0]))
        fx = f(x); steps += 1
    return x, fx, abs(fx / target - 1) <= tol, steps


def stage_n6(args, M, settings, cal: dict) -> None:
    """N6: for each method in --n6-methods, rescale the first --n6-k isotropic draws until their damage matches the
    method's at its replay request (x_push for push methods, x_shift for curves). Writes n6.json; the replay stage
    then runs each matched draw at its own strength as the null family n6:<method>."""
    out = Path(args.out)
    if (out / "n6.json").exists():
        return
    fac = EditFactory(args)
    md = method_dirs(Path(args.methods), args.site)
    dmg = Damage(M, fac.caa.layer)
    res, t0 = {}, time.time()
    for m in args.n6_methods:
        if m not in md:
            print(f"n6: no {m} bundle, skipped")
            continue
        meth = fac.method(str(md[m]))
        x_m = cal["x_shift"] if meth.meta["kind"] == "target" else cal["x_push"]
        target = dmg.pm(meth.edit, x_m, meth.positions)
        rec = {"x_method": x_m, "kl_method": target, "draws": []}
        if not target > 0:
            print(f"n6: {m} at x={x_m:g} does no measurable damage (KL {target:.2e}); no N6 family")
            res[m] = rec
            continue
        for i in range(args.n6_k):
            v = fac.draws["isotropic"][i].astype(np.float64)
            edit = fac.LinearEdit(v, fac.typical_norm)
            x, kl, ok, steps = match_strength(lambda z: dmg.pm(edit, z, fac.positions), target, max(cal["x_push"], 1e-3))
            rec["draws"].append({"row": i, "x": x, "kl": kl, "matched": bool(ok), "steps": steps})
        xs = [d["x"] for d in rec["draws"] if d["matched"]]
        print(f"n6: {m} at x={x_m:g}: KL {target:.4f}; {len(xs)}/{args.n6_k} draws matched, strength "
              f"{np.median(xs) if xs else float('nan'):.3g} (median) × typical norm  ({(time.time() - t0) / 60:.1f} min)")
        res[m] = rec
    write_json(out / "n6.json", {"tolerance": N6_TOLERANCE, "damage": "mean next-token KL(unedited || edited) over "
                                 f"{len(NEUTRAL_TEXTS)} neutral texts, mean of +x and -x", "neutral_texts": NEUTRAL_TEXTS,
                                 "draw_family": "isotropic", "methods": res})


def stage_replay(args, M, settings, rep: Replayer, cal: dict, split: str = "verification", keep_arm=None) -> None:
    """keep_arm(arm) -> bool picks the arms to run (default: every arm of build_arms)."""
    out = Path(args.out)
    adir = out / "arms"
    adir.mkdir(exist_ok=True)
    items = load_items(out, split)
    fac = EditFactory(args)
    # unsteered pass and the zero-edit check against a hook-free forward pass (N0)
    if not (adir / "_none.npz").exists():
        y0, m0 = rep.run(items, fac.caa.edit, 0.0, fac.positions)
        n0 = min(args.n0_check, len(items))
        alone = [float(rep.run([it], fac.caa.edit, 0.0, fac.positions)[0][0]) for it in items[:n0]]
        diffs = [a - float(rep.plain(it)[it["far_id"]] - rep.plain(it)[it["near_id"]]) for a, it in zip(alone, items)]
        # the same items inside a padded batch vs alone: the cost of batching, kept apart from the hook check above
        batch_diffs = [float(y0[i]) - a for i, a in enumerate(alone)]
        agrees = float(np.mean([(y > 0) == (it["chosen"] == "far") for y, it in zip(y0, items)]))
        savez_atomic(adir / "_none.npz", y=y0, mass=m0)
        write_json(out / "n0_check.json", {"n_checked": n0, "max_abs_diff_vs_hook_free": max(map(abs, diffs)) if diffs else None,
                                          "max_abs_diff_batched_vs_alone": max(map(abs, batch_diffs)) if batch_diffs else None,
                                          "replay_batch_size": rep.batch_size,
                                          "replay_matches_generated_choice": agrees})
        print(f"N0: zero edit vs hook-free max |diff| {max(map(abs, diffs)) if diffs else float('nan'):.3g}; "
              f"batched vs alone max |diff| {max(map(abs, batch_diffs)) if batch_diffs else float('nan'):.3g} "
              f"(batch {rep.batch_size}); unsteered replay picks the generated label on {agrees:.1%} of items")
    arms = [a for a in build_arms(args, cal) if keep_arm is None or keep_arm(a)]
    with open(out / "arms.jsonl", "w") as f:
        for a in arms:
            f.write(json.dumps(a) + "\n")
    t0 = time.time()
    for n, arm in enumerate(arms):
        path = adir / (hashlib.sha1(arm["arm"].encode()).hexdigest()[:12] + ".npz")
        if path.exists():
            continue
        twin = adir / (hashlib.sha1(arm.get("same_as", "").encode()).hexdigest()[:12] + ".npz")
        if arm.get("same_as") and twin.exists():
            savez_atomic(path, **dict(np.load(twin)))
            print(f"[{n + 1}/{len(arms)}] {arm['arm']} = {arm['same_as']} (same vector and strength)", flush=True)
            continue
        rec = {}
        if arm["family"] == "prompting":
            rec["y_plus"], rec["mass_plus"] = rep.run_prompted(items, PROMPTS[arm["plus"]])
            if not arm.get("single"):
                rec["y_minus"], rec["mass_minus"] = rep.run_prompted(items, PROMPTS[arm["minus"]])
        else:
            edit, positions = fac.make(arm)
            rec["y_plus"], rec["mass_plus"] = rep.run(items, edit, arm["x"], positions)
            if not arm.get("single"):
                rec["y_minus"], rec["mass_minus"] = rep.run(items, edit, -arm["x"], positions)
        savez_atomic(path, **rec)
        el = time.time() - t0
        print(f"[{n + 1}/{len(arms)}] {arm['arm']} x={arm['x']:g}  ({el / 60:.1f} min)", flush=True)


def stage_gencheck(args, M, settings, cal: dict) -> None:
    """Steered generation on a subset of verification pairs: none, and each of --gen-arms (caa +-x, isotropic[0] +-x,
    curve-displace +-shift)."""
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
    md = method_dirs(Path(args.methods), args.site)
    arms = [("none", None, 0.0)]
    for nm, spec in (("caa", {"family": "method", "src": str(md["caa"])}),
                     ("isotropic[0]", {"family": "isotropic", "row": 0}),
                     ("curve-displace", {"family": "method", "src": str(md.get("curve-displace", ""))})):
        if spec.get("src") == "" or nm not in args.gen_arms:
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
    p.add_argument("stages", nargs="+", choices=["items", "calibrate", "n6", "replay", "gencheck"])
    p.add_argument("--bank", required=True), p.add_argument("--run", required=True)
    p.add_argument("--activations", required=True), p.add_argument("--methods", required=True)
    p.add_argument("--draws", required=True), p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, default=sd.K_GEN, help="draws per null family (rows [:k] of the shared draws)")
    p.add_argument("--x-grid", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.2, 0.4, 0.8])
    p.add_argument("--target-logit", type=float, default=1.0)
    p.add_argument("--min-mass-ratio", type=float, default=0.95,
                   help="health bound: the even part of the label mass, (mass(+x) + mass(-x)) / 2, must stay >= this "
                        "fraction of the unsteered mass")
    p.add_argument("--extraction-index", help="extract.py's index.jsonl, to cross-check our label locator")
    p.add_argument("--x-shift", type=float, default=1.0, help="request for target methods, decades")
    p.add_argument("--calib-fold", type=int, default=0)
    p.add_argument("--n0-check", type=int, default=8)
    p.add_argument("--gen-pairs", type=int, default=16), p.add_argument("--gen-batch", type=int, default=8)
    p.add_argument("--replay-batch", type=int, default=16, help="prompts per teacher-forced replay pass")
    p.add_argument("--replay-dtype", default="float32", choices=["float32", "bfloat16"],
                   help="weights during calibrate and replay (weights_in); float32 takes 4 bytes per parameter "
                        "(an 8B model: 32 GB), so bfloat16 on a smaller GPU, at ~0.3 logit of rounding per item")
    p.add_argument("--prompting", action="store_true",
                   help="also replay the prompting baseline (PROMPTS: disposition and shift pairs, neutral sentence)")
    p.add_argument("--arms", nargs="+", help="replay only these arms, families or methods (e.g. caa curve-snap "
                                             "covariance); default every arm")
    p.add_argument("--gen-arms", nargs="+", default=["caa", "isotropic[0]", "curve-displace"],
                   help="steered arms of the generation check, each at +-x, beside the unsteered one")
    p.add_argument("--site", help="use only bundles fitted at this site (label, prelabel, ...); required when the "
                                  "methods directory holds a method at several sites")
    p.add_argument("--n6-methods", nargs="+", default=["caa", "curve-displace"],
                   help="methods that get a damage-matched null family (N6)")
    p.add_argument("--n6-k", type=int, default=sd.K_GEN, help="draws per N6 family (rows [:k] of the isotropic draws)")
    p.add_argument("--gen-samples", type=int, default=1,
                   help="generation check on a sampled baseline: passes per arm, seeds seed..seed+n-1 (greedy: 1)")
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    settings = run_settings(Path(args.run))
    write_json(out / "settings.json", {"args": vars(args), "run": settings})
    M = Model(settings["model_id"])
    from steer.methods import load
    layer = load(method_dirs(Path(args.methods), args.site)["caa"]).layer
    rep = Replayer(M, Path(args.run), Path(args.bank), settings["thinking"], layer, args.replay_batch)
    if "items" in args.stages:
        build_items(args, M, settings)
    cal = None
    # N6 measures damage in the replay's dtype, so the strengths it matches are the ones the replay then runs
    with weights_in(M.model, args.replay_dtype):
        if {"calibrate", "n6", "replay", "gencheck"} & set(args.stages):
            cal = stage_calibrate(args, M, settings, rep)
        if "n6" in args.stages:
            stage_n6(args, M, settings, cal)
        if "replay" in args.stages:
            stage_replay(args, M, settings, rep, cal, keep_arm=partial(arm_selected, wanted=args.arms))
    if "gencheck" in args.stages:
        stage_gencheck(args, M, settings, cal)


if __name__ == "__main__":
    main()
