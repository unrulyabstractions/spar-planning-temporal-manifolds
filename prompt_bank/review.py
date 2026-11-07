"""Step 3 of 3: a local web page for reviewing prompts, checker reports and baseline responses.

    python review.py --bank data/t0_s0               # then open http://127.0.0.1:5000
    python review.py --bank data/t0_s0 --port 5001 --runs-dir /path/to/runs

Run it after generate.py (prompts and checks) and baseline.py (responses). It shows:

* a table of every prompt with filters (split, regime, unit, checker disagreements, wrong or unparsed
  answers, your own review status, text search);
* for one prompt: its text and parameters, each checker's report, each baseline run's thinking, answer and
  parsed choice, and forms to record your judgement of the prompt and of each response.

Choose who you are in the drop-down at the top (there is no login). Judgements are appended to
<bank>/reviews/<your-name>.jsonl (pbank/annotations.py); commit that file to share your review.

Baseline runs are found automatically: every directory under --runs-dir (default: runs/) whose run.json was
made from this exact bank (same prompts SHA-256). Checker reports are read from <bank>/checks/.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

from flask import Flask, abort, redirect, render_template, request, url_for

from pbank.annotations import VERDICTS, add_annotation, current_annotations
from pbank.checker import latest_by_uid
from pbank.provenance import PACKAGE_DIR
from pbank.records import PromptRecord, load_prompts, prompts_sha256, read_jsonl
from pbank.t0 import CONSTRAINT_PHRASINGS

REVIEWER_COOKIE = "pbank_reviewer"


# ---------------------------------------------------------------------------------------------------------------
# Loading what there is to review
# ---------------------------------------------------------------------------------------------------------------

@dataclass
class Source:
    """A checker's reports or a baseline run's responses, keyed by prompt uid."""
    name: str                         # checker spec ("local:Qwen/Qwen3.5-27B") or run directory name
    kind: str                         # "checker" | "run"
    rows: dict[str, dict]             # uid -> latest row
    meta: dict = field(default_factory=dict)


def load_checkers(bank_dir: Path) -> list[Source]:
    out = []
    for path in sorted((bank_dir / "checks").glob("*.jsonl")):
        rows = latest_by_uid(read_jsonl(path))
        meta_path = path.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        name = meta.get("settings", {}).get("checker") or next(iter(rows.values()), {}).get("checker", path.stem)
        out.append(Source(name=name, kind="checker", rows=rows, meta=meta))
    return out


def load_runs(runs_dir: Path, bank_sha: str) -> list[Source]:
    """Every run under `runs_dir` made from the bank with this prompts hash."""
    out = []
    for run_json in sorted(runs_dir.glob("*/run.json")):
        meta = json.loads(run_json.read_text())
        if meta.get("settings", {}).get("bank_prompts_sha256") != bank_sha:
            continue
        rows = latest_by_uid(read_jsonl(run_json.parent / "responses.jsonl"))
        out.append(Source(name=run_json.parent.name, kind="run", rows=rows, meta=meta))
    return out


# ---------------------------------------------------------------------------------------------------------------
# What each prompt's row shows, and the filters
# ---------------------------------------------------------------------------------------------------------------

def checker_status(row: Optional[dict]) -> str:
    """'agree', 'disagree', 'unreadable' or 'missing' (not checked yet)."""
    if row is None:
        return "missing"
    if row.get("agrees") is None:
        return "unreadable"
    return "agree" if row["agrees"] else "disagree"


def run_status(row: Optional[dict]) -> str:
    """'correct', 'incorrect', 'answered' (no correct answer exists), 'unparsed', 'truncated' or 'missing'."""
    if row is None:
        return "missing"
    if row["hit_token_cap"]:
        return "truncated"
    if row["parse_status"] != "ok":
        return "unparsed"
    if row["correct"] is None:
        return "answered"
    return "correct" if row["correct"] else "incorrect"


FILTERS = {   # name -> (label, choices); the first choice means "no filter"
    "split": ("Split", ["all", "extraction", "verification"]),
    "regime": ("Regime", ["all", "none", "near_only", "both", "no_horizon"]),
    "unit": ("Horizon unit", ["all", "seconds", "minutes", "hours", "days", "weeks", "months", "years", "none"]),
    "rewritten": ("Unit rewrite", ["all", "yes", "no"]),
    "checker": ("Checker", ["all", "any issue", "disagree", "flagged a problem", "unreadable"]),
    "run": ("Baseline", ["all", "any issue", "incorrect", "unparsed", "truncated"]),
    "review": ("My review", ["all", "not reviewed by me", "reviewed by me", "marked problem by anyone"]),
}


def review_summary(notes: dict, me: str) -> dict[str, dict]:
    """Per prompt uid: how many judgements it has (of the prompt and of responses), whether any says "problem",
    and the current reviewer's own judgement of the prompt."""
    out: dict[str, dict] = {}
    for (uid, target), by in notes.items():
        s = out.setdefault(uid, {"count": 0, "problem": False, "mine": None, "mine_any": False})
        for who, note in by.items():
            s["count"] += 1
            s["problem"] |= note["verdict"] == "problem"
            if who == me:
                s["mine_any"] = True
                if target == "prompt":
                    s["mine"] = note["verdict"]
    return out


def matches(rec: PromptRecord, f: dict, checkers: list[Source], runs: list[Source], reviews: dict) -> bool:
    """Whether a prompt passes every active filter in `f`."""
    if f["split"] != "all" and rec.split != f["split"]:
        return False
    if f["regime"] != "all" and rec.regime != f["regime"]:
        return False
    if f["unit"] != "all" and (rec.horizon_unit or "none") != f["unit"]:
        return False
    if f["rewritten"] != "all" and rec.rewritten != (f["rewritten"] == "yes"):
        return False
    if f["checker"] != "all":
        rows = [c.rows.get(rec.uid) for c in checkers]
        found = {"disagree": any(checker_status(r) == "disagree" for r in rows),
                 "unreadable": any(checker_status(r) == "unreadable" for r in rows),
                 "flagged a problem": any(r and r.get("verdict") == "problem" for r in rows)}
        if not (any(found.values()) if f["checker"] == "any issue" else found[f["checker"]]):
            return False
    if f["run"] != "all":
        statuses = {run_status(r.rows.get(rec.uid)) for r in runs}
        wanted = {"incorrect", "unparsed", "truncated"} if f["run"] == "any issue" else {f["run"]}
        if not statuses & wanted:
            return False
    if f["review"] != "all":
        s = reviews.get(rec.uid, {})
        ok = {"not reviewed by me": not s.get("mine_any"), "reviewed by me": bool(s.get("mine_any")),
              "marked problem by anyone": bool(s.get("problem"))}[f["review"]]
        if not ok:
            return False
    q = f.get("q", "").strip().lower()
    return not q or q in rec.uid.lower() or q in rec.text.lower()


# ---------------------------------------------------------------------------------------------------------------
# The app
# ---------------------------------------------------------------------------------------------------------------

def create_app(bank_dir: Path, runs_dir: Path, reviewers: list[str]) -> Flask:
    """Load the bank once; annotations are re-read on every request, since reviewers add them as they go."""
    bank_dir = Path(bank_dir)
    records = load_prompts(bank_dir)
    if not records:
        raise SystemExit(f"no prompts in {bank_dir}; run generate.py first")
    by_uid = {r.uid: r for r in records}
    bank_sha = prompts_sha256(records)
    checkers = load_checkers(bank_dir)
    runs = load_runs(runs_dir, bank_sha)

    app = Flask(__name__, template_folder=str(PACKAGE_DIR / "templates"))

    def current_filters() -> dict:
        f = {name: request.args.get(name, choices[0]) for name, (_, choices) in FILTERS.items()}
        f["q"] = request.args.get("q", "")
        return f

    def filtered_uids(f: dict, reviews: dict) -> list[str]:
        return [r.uid for r in records if matches(r, f, checkers, runs, reviews)]

    @app.context_processor
    def common():
        return {"reviewers": reviewers, "me": request.cookies.get(REVIEWER_COOKIE, ""), "bank_name": bank_dir.name,
                "checkers": checkers, "runs": runs, "checker_status": checker_status, "run_status": run_status}

    @app.get("/")
    def index():
        me = request.cookies.get(REVIEWER_COOKIE, "")
        reviews = review_summary(current_annotations(bank_dir), me)
        f = current_filters()
        uids = filtered_uids(f, reviews)
        return render_template("index.html", rows=[by_uid[u] for u in uids], total=len(records), filters=f,
                               filter_defs=FILTERS, reviews=reviews, query=urlencode(f))

    @app.get("/item/<uid>")
    def item(uid: str):
        rec = by_uid.get(uid) or abort(404)
        me = request.cookies.get(REVIEWER_COOKIE, "")
        notes = current_annotations(bank_dir)
        f = current_filters()
        uids = filtered_uids(f, review_summary(notes, me))
        pos = uids.index(uid) if uid in uids else None
        neighbours = {"prev": uids[pos - 1] if pos else None,
                      "next": uids[pos + 1] if pos is not None and pos + 1 < len(uids) else None}
        phrasing = None if rec.phrasing_id is None else CONSTRAINT_PHRASINGS[rec.phrasing_id]
        return render_template("item.html", rec=rec, notes=notes, verdicts=VERDICTS, neighbours=neighbours,
                               position=None if pos is None else pos + 1, n_filtered=len(uids),
                               query=urlencode(f), phrasing=phrasing)

    @app.post("/annotate")
    def annotate():
        me = request.cookies.get(REVIEWER_COOKIE, "")
        if me not in reviewers:
            abort(400, "choose who you are in the drop-down at the top of the page first")
        uid, target = request.form["uid"], request.form["target"]
        if uid not in by_uid or target not in ["prompt"] + [r.name for r in runs]:
            abort(400, "unknown prompt or target")
        add_annotation(bank_dir, me, uid, target, request.form["verdict"], request.form.get("note", ""))
        return redirect(url_for("item", uid=uid) + "?" + request.form.get("query", "") + f"#{target}")

    @app.post("/reviewer")
    def set_reviewer():
        name = request.form.get("reviewer", "")
        response = redirect(request.form.get("back") or url_for("index"))
        if name in reviewers:
            response.set_cookie(REVIEWER_COOKIE, name, max_age=365 * 24 * 3600)
        else:
            response.delete_cookie(REVIEWER_COOKIE)
        return response

    return app


def read_reviewers(path: Path) -> list[str]:
    """One name per line; blank lines and lines starting with # are ignored."""
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bank", type=Path, required=True, help="bank directory written by generate.py")
    p.add_argument("--runs-dir", type=Path, default=PACKAGE_DIR / "runs", help="where baseline.py wrote its runs")
    p.add_argument("--reviewers", type=Path, default=PACKAGE_DIR / "reviewers.txt")
    p.add_argument("--port", type=int, default=5000)
    args = p.parse_args()
    app = create_app(args.bank, args.runs_dir, read_reviewers(args.reviewers))
    # 127.0.0.1 only: the page writes files and has no login, so it must not be reachable from other machines.
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
