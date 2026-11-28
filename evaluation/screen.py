"""Screen of steering configs (bundle × write span) on held-out horizons, and selection of the one to confirm.

  screen  For each bundle (method at its own layer and site) and write span: sweep the strength grid (push: x ×
          typical norm; target: shift in decades) at +s and -s on the screen items: extraction-split prompts whose uid
          has --uids-role in --uids, scored regimes only, readable in --items-from. Each strength is a calibration
          step of replay_eval (sweep_point: order-balanced mean odd effect, label-mass health). The sweep stops at the
          first unhealthy strength or the first sign change versus the first strength. S = choose_strength's "peak"
          rule. One JSON line per config in screen.jsonl, written as each finishes; a restart skips configs already
          there. No config starts past --deadline.
          Patch baselines (--patch-bundles, --patch-layers; patching.py), never candidates: per horizon write span,
          the full patch at each --patch-layers layer and a subspace patch per --patch-bundles bundle (its direction or
          curve PCs, at its layer), at x = ±1 (source horizon 10x longer / shorter); and the text rewrite once (span
          "text"). One point each, scored like a strength of the sweep.
  select  The winner: the highest S among candidates (not the controls choice and reward) whose chosen strength is
          healthy; S within TIE_LOGITS of the best goes to the smaller strength. Writes winner.json, screen.md and, in
          --confirm-out, the calibration.json that fixes replay_eval's strength for the confirm run.

Usage:
  python screen.py screen --bank B --run R --items-from OUT --uids splits/reserved_horizons.jsonl --uids-role test \\
      --bundles METHODS/caa__... METHODS/trend__... --write-span all prompt --out SCREEN [--tf32] [--deadline UNIX]
  python screen.py select --out SCREEN --confirm-out CONFIRM
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import patching
import replay_eval as rv

TF32_MAX_DIFF = 0.05      # logits: TF32 is used only if unsteered y moves by at most this on every screen item
TIE_LOGITS = 0.25
SELECTION_RULE = (f"highest S (peak-rule order-balanced mean odd effect, logits toward far) among methods other than "
                  f"{' and '.join(rv.CONTROLS)} whose chosen strength is healthy; S within {TIE_LOGITS} logits of the "
                  f"best: the smallest strength, compared by position in its own grid, then the higher S")


# ------------------------------------------------------------------ screen

def patch_configs(bundles: list, layers: list) -> dict:
    """Patch baseline configs, keyed by a pseudo bundle path whose name keys screen.jsonl: name -> (what, where)."""
    out = {Path("patch:text"): ("text", None)}
    out.update({Path(f"patch:full@L{L}"): ("full", int(L)) for L in layers})
    out.update({Path(f"patch:{Path(b).name}"): ("basis", Path(b)) for b in bundles})
    return out


def is_baseline(method: str) -> bool:
    """Controls and patch baselines are screened beside the candidates but never selected."""
    return method in rv.CONTROLS or method.startswith("patch:")


def screen_items(items_from: Path, uids: Path, role: str) -> list[dict]:
    keep = {r["uid"] for r in rv.read_jsonl(uids) if r["role"] == role}
    return [it for it in rv.load_items(items_from, "extraction") if it["uid"] in keep and it["regime"] in rv.SCORED]


def screen_row(rep, items, edit, s: float, positions, mass0: float) -> dict:
    """replay_eval's calibration step at strength s, plus the odd effect by regime and by option order."""
    row, o = rv.sweep_point(rep, items, edit, s, positions, mass0, rv.MIN_MASS_RATIO)
    regime = np.array([it["regime"] for it in items])
    for r in rv.SCORED:
        sel = regime == r
        row[f"odd_{r}"] = rv.order_balanced_mean(o[sel], [it for it, k in zip(items, sel) if k]) if sel.any() else None
    nf = np.array([it["near_first"] for it in items])
    far_first = float(o[~nf].mean()) if (~nf).any() else None
    near_first = float(o[nf].mean()) if nf.any() else None
    row["odd_far_first"], row["odd_near_first"] = far_first, near_first
    # push toward the first-listed label (analyze.py's letter part, from the two orders' means: items are not paired)
    row["letter"] = (far_first - near_first) / 2 if far_first is not None and near_first is not None else None
    return row


def sweep_until_stop(point, grid) -> list[dict]:
    """point(s) -> sweep row for s along the grid, stopping after the first unhealthy row or the first row whose effect
    has another sign than the first row's: past either, the strength is not the method's own effect."""
    rows = []
    for s in grid:
        rows.append(point(s))
        if not rows[-1]["healthy"] or np.sign(rows[-1]["mean_odd_logit"]) != np.sign(rows[0]["mean_odd_logit"]):
            break
    return rows


def screen_config(point, grid, mass0: float) -> dict:
    sweep = sweep_until_stop(point, grid)
    strength, rule = rv.choose_strength(sweep, "peak", target_logit=None)   # the peak rule has no target
    at = next(row for row in sweep if row["x"] == strength)
    return {"grid": list(grid), "sweep": sweep, "strength": strength, "strength_rank": list(grid).index(strength),
            "rule": rule, "S": at["mean_odd_logit"], "healthy": at["healthy"],
            "mass_ratio": at["even_label_mass"] / mass0, "unsteered_label_mass": mass0,
            **{k: at[k] for k in ("odd_near_only", "odd_both", "odd_far_first", "odd_near_first", "letter")}}


def done_configs(path: Path) -> set:
    return {(r["bundle"], r["span"]) for r in rv.read_jsonl(path)} if path.exists() else set()


def run_screen(todo: list, run_config, path: Path, deadline: float | None) -> None:
    """run_config(bundle, span) -> record, for each (bundle dir, span) of todo; each record is appended to path as
    soon as it is done, so a restart loses at most the config in progress."""
    t0 = time.time()
    for n, (bundle, span) in enumerate(todo):
        if deadline is not None and time.time() > deadline:
            print(f"screen: deadline passed, {len(todo) - n} configs not started", flush=True)
            return
        start = time.time()
        rec = {"bundle": bundle.name, "span": span, **run_config(bundle, span)}
        rec["seconds"] = time.time() - start
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        elapsed = time.time() - t0
        eta = elapsed / (n + 1) * (len(todo) - n - 1)
        print(f"[{n + 1}/{len(todo)}] {bundle.name} span={span}: S {rec['S']:+.3f} at {rec['strength']:g}"
              f"{'' if rec['healthy'] else ' (unhealthy)'}, {rec['seconds'] / 60:.1f} min; elapsed "
              f"{elapsed / 60:.1f} min, ETA {eta / 60:.1f} min", flush=True)


def tf32_check(rep, items):
    """Unsteered y with TF32 off and on. TF32 stays on only if no item moves by more than TF32_MAX_DIFF.
    Returns (the check's record, (y, mass) unsteered in the mode kept)."""
    rv.set_tf32(False)
    y_fp32, m_fp32 = rep.unsteered(items)
    rv.set_tf32(True)
    y_tf32, m_tf32 = rep.unsteered(items)
    diff = float(np.abs(y_tf32 - y_fp32).max())
    used = diff <= TF32_MAX_DIFF
    rv.set_tf32(used)
    print(f"TF32 check: max |y(TF32) - y(fp32)| = {diff:.4f} logits on {len(items)} items: "
          f"{'TF32 on' if used else f'over {TF32_MAX_DIFF}, TF32 off for the whole screen'}", flush=True)
    check = {"max_abs_diff_logits": diff, "threshold_logits": TF32_MAX_DIFF, "used": used, "n_items": len(items)}
    return check, ((y_tf32, m_tf32) if used else (y_fp32, m_fp32))


def cmd_screen(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "screen.jsonl"
    bundles = [Path(b) for b in args.bundles or []]
    if len({b.name for b in bundles}) != len(bundles):
        sys.exit("bundle directory names must be unique: they key screen.jsonl")
    patch_bundles, patch_layers = getattr(args, "patch_bundles", None) or [], getattr(args, "patch_layers", None) or []
    patches = patch_configs(patch_bundles, patch_layers) if (patch_bundles or patch_layers) else {}
    patch_spans = [span for span in args.write_span if span in patching.PATCH_SPANS]
    if patches and not patch_spans:
        sys.exit("patch baselines need --write-span horizon and/or from_horizon")
    done = done_configs(path)
    wanted = [(b, span) for b in bundles for span in args.write_span]
    wanted += [(n, span) for n, (what, _) in patches.items() for span in (["text"] if what == "text" else patch_spans)]
    todo = [(b, span) for b, span in wanted if (b.name, span) not in done]
    print(f"screen: {len(todo)} configs to run, {len(wanted) - len(todo)} already in {path}")
    if not todo:
        return
    from steer.methods import load
    settings = rv.run_settings(Path(args.run))
    items = screen_items(Path(args.items_from), Path(args.uids), args.uids_role)
    if not items:
        sys.exit(f"no readable scored extraction items with role {args.uids_role!r}")
    M = rv.Model(settings["model_id"])
    first = next((b for b, _ in todo if b not in patches), None)
    layer = load(first).layer if first else next(w if what == "full" else load(w).layer
                                                 for what, w in patches.values() if what != "text")
    rep = rv.Replayer(M, Path(args.run), Path(args.bank), settings["thinking"], layer, args.replay_batch)
    with rv.weights_in(M.model, args.replay_dtype):
        if args.tf32:
            check, (y0, m0) = tf32_check(rep, items)
        else:
            check, (y0, m0) = None, rep.unsteered(items)
        tf32 = bool(check and check["used"])
        mass0 = float(m0.mean())
        print(f"screen: {len(items)} items, unsteered label mass {mass0:.3f}", flush=True)
        # one entry per invocation: a screen may be run in parts (other bundles or spans) into the same --out
        sp = out / "screen_settings.json"
        runs = json.loads(sp.read_text())["runs"] if sp.exists() else []
        runs.append({"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "args": vars(args),
                     "run": settings, "n_items": len(items), "tf32_check": check, "tf32": tf32,
                     "min_mass_ratio": rv.MIN_MASS_RATIO, "unsteered_label_mass": mass0,
                     "unsteered": {it["uid"]: {"y": float(y), "mass": float(m)} for it, y, m in zip(items, y0, m0)}})
        rv.write_json(sp, {"runs": runs})

        def run_patch(name: Path, span: str) -> dict:
            what, where = patches[name]
            if what == "text":
                edit, site = patching.TextEdit(), "horizon"
            elif what == "full":
                edit, site, rep.layer = patching.PatchEdit(), "horizon_last", where
            else:
                meth = load(where)
                edit, site, rep.layer = patching.PatchEdit(patching.method_basis(meth)), meth.site, meth.layer
            rep.write_span = span if span in patching.PATCH_SPANS else "all"
            row = screen_row(rep, items, edit, 1.0, "all", mass0)
            print(f"  {name.name} span={span}: odd {row['mean_odd_logit']:+.3f}, even label mass "
                  f"{row['even_label_mass']:.3f}", flush=True)
            return {"method": name.name.split("@")[0] if what != "basis" else f"patch:{load(where).meta['method']}",
                    "layer": None if what == "text" else rep.layer, "site": site, "kind": "patch",
                    "path": str(where) if what == "basis" else None, "n_items": len(items),
                    "replay_dtype": args.replay_dtype, "tf32": tf32, **screen_config(lambda s: row, [1.0], mass0)}

        def run_config(bundle: Path, span: str) -> dict:
            if bundle in patches:
                return run_patch(bundle, span)
            meth = load(bundle)
            rep.layer, rep.write_span = meth.layer, span
            grid = args.push_grid if meth.kind == "push" else args.shift_grid

            def point(s):
                row = screen_row(rep, items, meth.edit, s, meth.positions, mass0)
                print(f"  {bundle.name} span={span} s={s:g}: odd {row['mean_odd_logit']:+.3f}, even label mass "
                      f"{row['even_label_mass']:.3f}{'' if row['healthy'] else ' UNHEALTHY'}", flush=True)
                return row

            return {"method": meth.meta["method"], "layer": meth.layer, "site": meth.site, "kind": meth.kind,
                    "path": str(bundle), "n_items": len(items), "replay_dtype": args.replay_dtype, "tf32": tf32,
                    **screen_config(point, grid, mass0)}

        run_screen(todo, run_config, path, args.deadline)


# ------------------------------------------------------------------ select

def select(records: list[dict]):
    """(winner or None, candidates sorted by S, highest first)."""
    candidates = sorted((r for r in records if not is_baseline(r["method"]) and r["healthy"]), key=lambda r: -r["S"])
    if not candidates:
        return None, []
    tied = [r for r in candidates if r["S"] >= candidates[0]["S"] - TIE_LOGITS]
    return min(tied, key=lambda r: (r["strength_rank"], -r["S"])), candidates


def status(r: dict, winner) -> str:
    if r is winner:
        return "winner"
    if r["method"].startswith("patch:"):
        return "patch baseline"
    if r["method"] in rv.CONTROLS:
        return "control"
    return "candidate" if r["healthy"] else "unhealthy"


def fmt(v, spec="+.3f"):
    return "–" if v is None else format(v, spec)


def screen_md(records: list[dict], winner) -> str:
    L = ["# Screen: held-out horizons, replay readout", "", f"Rule: {SELECTION_RULE}.", "",
         f"Winner: {winner['bundle']} (span {winner['span']}), S = {winner['S']:+.3f} at {winner['strength']:g}"
         if winner else "No winner: no candidate has a healthy strength.", "",
         "S and odd: order-balanced mean odd effect at the chosen strength (logits, + = toward far). Strength: x × "
         "typical norm (push) or Δ decades (target). Mass ratio: even label mass / unsteered.", "",
         "| layer | site | method | span | strength | S | mass ratio | odd near_only | odd both | letter | status |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(records, key=lambda r: -r["S"]):
        strength = {"push": f"x {r['strength']:g}", "patch": "source ×10^±1"}.get(r["kind"], f"Δ {r['strength']:g}")
        L.append(f"| {r['layer']} | {r['site']} | {r['method']} | {r['span']} | {strength} | {r['S']:+.3f} | "
                 f"{r['mass_ratio']:.3f} | {fmt(r['odd_near_only'])} | {fmt(r['odd_both'])} | {fmt(r['letter'])} | "
                 f"{status(r, winner)} |")
    return "\n".join(L) + "\n"


PATCH_RULE = ("the layer and span of the patch:full row with the largest S; there, the patch subspace (patch:<method> "
              "row) with the largest S")


def best_patch(records: list[dict]) -> dict | None:
    """Which patch the confirm stage replays (PATCH_RULE), or None when the screen has no full patch."""
    full = [r for r in records if r["method"] == "patch:full"]
    if not full:
        return None
    top = max(full, key=lambda r: r["S"])
    bases = [r for r in records if r["method"].startswith("patch:") and r["method"] not in ("patch:full", "patch:text")
             and r["layer"] == top["layer"] and r["span"] == top["span"]]
    basis = max(bases, key=lambda r: r["S"]) if bases else None
    return {"rule": PATCH_RULE, "layer": top["layer"], "span": top["span"], "full_S": top["S"],
            "basis": None if basis is None else {k: basis[k] for k in ("bundle", "path", "method", "site", "S")}}


def patch_calibration(span: str) -> dict:
    """replay_eval's calibration.json for a patch confirm run: no strength to calibrate (x = ±1, the source prompts)."""
    return {"fold": None, "n_train": None, "n_heldout_scored": None, "target_logit": None, "min_mass_ratio": None,
            "sweep": [], "x_push": None, "x_shift": 1.0, "reached_target": None, "write_span": span,
            "rule": "patch confirm: x = ±1 is the source prompt 10x longer / shorter; nothing calibrated"}


def confirm_calibration(w: dict) -> dict:
    """replay_eval's calibration.json with the winner's strength: its calibrate stage returns this file unchanged, so
    the confirm run replays at the screened strength and span."""
    push = w["kind"] == "push"
    return {"fold": None, "n_train": None, "n_heldout_scored": w["n_items"], "target_logit": None,
            "min_mass_ratio": rv.MIN_MASS_RATIO, "unsteered_label_mass": w["unsteered_label_mass"],
            "sweep": w["sweep"],
            # only the winner's kind is screened: a curve winner has no push strength, and a push winner gets
            # replay_eval's --x-shift default for any curve arm beside it
            "x_push": w["strength"] if push else None, "x_shift": 1.0 if push else w["strength"],
            "rule": f"fixed by screen.py select: {SELECTION_RULE}", "reached_target": None, "write_span": w["span"],
            "screen_winner": {k: w[k] for k in ("bundle", "path", "method", "layer", "site", "span", "kind",
                                                "strength", "S")}}


def cmd_select(args) -> None:
    out = Path(args.out)
    records = rv.read_jsonl(out / "screen.jsonl")
    winner, candidates = select(records)
    (out / "screen.md").write_text(screen_md(records, winner))
    table = [{k: r[k] for k in ("bundle", "method", "layer", "site", "span", "kind", "strength", "S", "mass_ratio")}
             for r in candidates]
    rv.write_json(out / "winner.json", {"rule": SELECTION_RULE, "winner": winner, "candidates": table,
                                        "n_configs": len(records)})
    if winner is None:
        sys.exit("select: no candidate with a healthy strength; no calibration.json written")
    print(f"select: {winner['bundle']} span={winner['span']}, S {winner['S']:+.3f} at {winner['strength']:g}")
    confirm = Path(args.confirm_out)
    confirm.mkdir(parents=True, exist_ok=True)
    cal = confirm_calibration(winner)
    cal_path = confirm / "calibration.json"
    if cal_path.exists() and json.loads(cal_path.read_text()) != json.loads(json.dumps(cal, default=str)):
        sys.exit(f"{cal_path} exists with another strength or winner; remove it to select again")
    rv.write_json(cal_path, cal)


def cmd_patch_select(args) -> None:
    choice = best_patch(rv.read_jsonl(Path(args.out) / "screen.jsonl"))
    if choice is None:
        sys.exit("patch-select: the screen has no patch:full rows")
    rv.write_json(Path(args.out) / "patch_choice.json", choice)
    confirm = Path(args.confirm_out)
    confirm.mkdir(parents=True, exist_ok=True)
    rv.write_json(confirm / "calibration.json", patch_calibration(choice["span"]))
    print(f"patch-select: L{choice['layer']} span {choice['span']} (full S {choice['full_S']:+.3f}); basis "
          f"{choice['basis'] and choice['basis']['bundle']}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("screen", help="sweep every bundle × write span on the screen items")
    s.add_argument("--bank", required=True), s.add_argument("--run", required=True)
    s.add_argument("--activations", help="recorded in screen_settings.json only (the screen refits nothing)")
    s.add_argument("--out", required=True)
    s.add_argument("--items-from", required=True, help="a replay_eval out dir whose items stage ran (items.jsonl)")
    s.add_argument("--uids", required=True, help="jsonl with uid and role, e.g. splits/reserved_horizons.jsonl")
    s.add_argument("--uids-role", default="test")
    s.add_argument("--bundles", nargs="+", default=[], help="method bundle directories (method.json each)")
    s.add_argument("--patch-bundles", nargs="+", default=[],
                   help="bundles whose direction or curve PCs are patch subspaces (a baseline per bundle and span)")
    s.add_argument("--patch-layers", nargs="+", type=int, default=[], help="layers of the full patch baseline")
    s.add_argument("--write-span", nargs="+", choices=rv.WRITE_SPANS, default=["all"])
    s.add_argument("--replay-dtype", default="float32", choices=["float32", "bfloat16"])
    s.add_argument("--tf32", action="store_true",
                   help=f"TF32 matmuls, if unsteered y moves by <= {TF32_MAX_DIFF} logits on every screen item")
    s.add_argument("--replay-batch", type=int, default=16, help="prompts per teacher-forced replay pass")
    s.add_argument("--deadline", type=float, help="unix time: no config starts after it")
    s.add_argument("--push-grid", type=float, nargs="+", default=[0.1, 0.2, 0.4, 0.8], help="x × typical norm")
    s.add_argument("--shift-grid", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0], help="Δ, decades")
    c = sub.add_parser("select", help="pick the winner from screen.jsonl")
    c.add_argument("--out", required=True, help="the screen's --out (screen.jsonl)")
    c.add_argument("--confirm-out", required=True, help="the confirm run's replay_eval --out: gets calibration.json")
    q = sub.add_parser("patch-select", help="pick the patch to confirm from screen.jsonl (PATCH_RULE)")
    q.add_argument("--out", required=True, help="the screen's --out (screen.jsonl); gets patch_choice.json")
    q.add_argument("--confirm-out", required=True, help="the patch confirm run's --out: gets calibration.json")
    args = p.parse_args(argv)
    {"screen": cmd_screen, "select": cmd_select, "patch-select": cmd_patch_select}[args.cmd](args)


if __name__ == "__main__":
    main()
