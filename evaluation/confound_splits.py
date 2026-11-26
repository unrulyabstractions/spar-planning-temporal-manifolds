"""Splits of the T0 bank that diagnose confounds, a balance table for them, and the minimal-pair bank.

    python confound_splits.py splits --bank ../prompt_bank/data/t0_s0           # index files + balance.md/json
    python confound_splits.py minimal-pairs --out ../prompt_bank/data/t0_mp0   # the minimal-pair bank

A split is an index file `<name>.jsonl` of {"uid", "role", <why>} rows plus `<name>.manifest.json` (name, rule,
seed, bank hash, counts). It only selects uids of the existing bank; no prompt, label or bank file changes.
Roles: "fit" and "test", and "drop" in `debatable` (records to leave out of fitting).

    reserved_horizons  C8: 8 of the 48 horizon grid values are test, the rest fit (extraction split)
    c5_surface         C5: fit on phrasings {0, 2}, original wording, a)/b); each test set changes one of those
    c1_balanced        C1: per CAA class, the far label's position and the label style balanced in size
    debatable          the exact ties and the unit rewrites within 1% of a near delay (checker_triage_t0.md)

`balance_table(records, factors, by)` crosstabs nuisance factors against regime and correct option, and flags
cells that deviate from the overall share. `build_minimal_pairs` makes the bank of Task "minimal pairs": the same
item just below and just above the far delay, in both option orders.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "prompt_bank"))
from label_audit import debatable_kinds, parse_durations, table  # noqa: E402
from pbank.durations import HORIZON_GRID  # noqa: E402
from pbank.provenance import git_commit, utc_now  # noqa: E402
from pbank.records import prompts_sha256, read_jsonl, record_to_dict, write_jsonl  # noqa: E402
from pbank.t0 import (CONSTRAINT_PHRASINGS, FAR_DELAYS, LABEL_STYLES, NEAR_DELAYS, NEAR_REWARDS,  # noqa: E402
                      REWARD_MULTIPLIERS, horizon_in_years, make_record, require_distinct_texts)

SEED = 0
HELD_OUT_GRID_INDICES = range(2, len(HORIZON_GRID), 6)   # 3rd, 9th, ... of 48: 8 values, both ends stay in fitting
FIT_PHRASINGS = (0, 2)
LONG_AT_LEAST, SHORT_AT_MOST = 4, 0.25                    # CAA classes by horizon / far delay
BALANCE_FACTORS = ("labels", "near_first", "phrasing_id", "rewritten", "reward_multiplier")
BALANCE_BY = ("regime", "correct_option")
FLAG_DEVIATION, MIN_GROUP = 0.10, 10


# ---------------------------------------------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------------------------------------------

def extraction_with_horizon(records: list[dict]) -> list[dict]:
    return [r for r in records if r["split"] == "extraction" and r["horizon"] is not None]


def reserved_horizons(records: list[dict]) -> list[dict]:
    """C8: prompts whose horizon grid value is held out are test, the others fit. A null twin has no horizon, so
    it follows its source prompt (same options, same order): it would leak the held-out prompt into fitting."""
    held_out = {str(HORIZON_GRID[i]) for i in HELD_OUT_GRID_INDICES}
    source_grid_value = {r["uid"]: r["horizon_grid_value"] for r in records}
    rows = []
    for r in (r for r in records if r["split"] == "extraction"):
        grid_value = r["horizon_grid_value"] or source_grid_value[r["uid"].removesuffix("-null")]
        rows.append({"uid": r["uid"], "role": "test" if grid_value in held_out else "fit",
                     "horizon_grid_value": grid_value})
    return rows


def c5_surface(records: list[dict]) -> list[dict]:
    """C5: fit = phrasings {0, 2}, grid-unit wording, a)/b). Each test set changes exactly one of those three
    and keeps the other two, so a drop on it is attributable to that one surface feature."""
    def role(r):
        in_fit_phrasing = r["phrasing_id"] in FIT_PHRASINGS
        original, ab = not r["rewritten"], r["labels"][0] == "a)"
        if in_fit_phrasing and original and ab:
            return "fit"
        if not in_fit_phrasing and original and ab:
            return "test_phrasing"
        if in_fit_phrasing and not original and ab:
            return "test_rewrite"
        if in_fit_phrasing and original and not ab:
            return "test_labels"
        return None   # differs in two or more features: in no set

    rows = []
    for r in extraction_with_horizon(records):
        if role(r):
            rows.append({"uid": r["uid"], "role": "fit" if role(r) == "fit" else "test", "test_set": role(r),
                         "phrasing_id": r["phrasing_id"], "rewritten": r["rewritten"], "labels": "/".join(r["labels"])})
    return rows


def caa_class(r: dict) -> str | None:
    if r["h_over_far_delay"] >= LONG_AT_LEAST:
        return "long"
    return "short" if r["h_over_far_delay"] <= SHORT_AT_MOST else None


def c1_balanced(records: list[dict], include_none: bool, seed: int = SEED) -> list[dict]:
    """C1: per CAA class (long: H/far delay >= 4, short: <= 0.25), the same number of prompts in each cell of
    near_first x label style. near_first fixes which option is listed first and so whether the far option
    carries the first or the second label; the style fixes the label's characters ("a)"/"1)" vs "b)"/"2)").
    Every cell of both classes is cut to the smallest cell (seeded sample), so the classes are as large as
    possible while equal in size and balanced. `none`-regime prompts (no option pays off) are out unless asked."""
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for r in extraction_with_horizon(records):
        cls = caa_class(r)
        if cls and (include_none or r["regime"] != "none"):
            cells[(cls, r["near_first"], r["labels"][0])].append(r)
    keys = [(cls, nf, style) for cls in ("long", "short") for nf in (True, False) for style in ("a)", "1)")]
    size = min(len(cells[k]) for k in keys)
    rng = random.Random(f"{seed}-c1-{'with-none' if include_none else 'no-none'}")
    rows = []
    for key in keys:
        for r in sorted(rng.sample(sorted(cells[key], key=lambda r: r["uid"]), size), key=lambda r: r["uid"]):
            rows.append({"uid": r["uid"], "role": "fit", "caa_class": key[0], "near_first": key[1],
                         "far_label": r["labels"][1] if r["near_first"] else r["labels"][0],
                         "regime": r["regime"]})
    return rows


def debatable(records: list[dict]) -> list[dict]:
    """Exact ties, and unit rewrites within 1% of a delay (the four "183 days" vs "6 months" prompts, 0.2% over)."""
    rows = []
    for r in records:
        kinds = debatable_kinds(r, parse_durations(r))
        if "tie" in kinds or {"near_boundary", "rewrite"} <= set(kinds):
            rows.append({"uid": r["uid"], "role": "drop", "why": "+".join(kinds)})
    return rows


SPLIT_RULES = {
    "reserved_horizons": "extraction split; horizon grid values 3, 9, ..., 45 of the 48 are test, the rest fit; "
                         "null twins follow their source prompt",
    "c5_surface": "extraction split with a horizon; fit = phrasings {0, 2}, not rewritten, a)/b); test_phrasing = "
                  "phrasings {1, 3}; test_rewrite = rewritten; test_labels = 1)/2); others (two features differ) in no set",
    "c1_balanced": "extraction split, CAA classes long (H/far delay >= 4) and short (<= 0.25), `none` regime left out; "
                   "every near_first x label-style cell of both classes cut to the smallest cell",
    "c1_balanced_with_none": "as c1_balanced, `none`-regime prompts included",
    "debatable": "all splits; exact ties (horizon equals a delay) and unit rewrites within 1% of a delay; role drop",
}


def build_splits(records: list[dict], seed: int = SEED) -> dict[str, list[dict]]:
    return {
        "reserved_horizons": reserved_horizons(records),
        "c5_surface": c5_surface(records),
        "c1_balanced": c1_balanced(records, include_none=False, seed=seed),
        "c1_balanced_with_none": c1_balanced(records, include_none=True, seed=seed),
        "debatable": debatable(records),
    }


def split_manifest(name: str, rows: list[dict], bank_sha256: str, seed: int) -> dict:
    return {"name": name, "rule": SPLIT_RULES[name], "seed": seed, "bank_prompts_sha256": bank_sha256,
            "counts": {"rows": len(rows), "role": dict(Counter(r["role"] for r in rows)),
                       **({"test_set": dict(Counter(r["test_set"] for r in rows))} if name == "c5_surface" else {}),
                       **({"caa_class": dict(Counter(r["caa_class"] for r in rows))} if name.startswith("c1") else {})}}


def rows_of_split(records: list[dict], rows: list[dict], role: str) -> list[dict]:
    wanted = {r["uid"] for r in rows if r["role"] == role}
    return [r for r in records if r["uid"] in wanted]


# ---------------------------------------------------------------------------------------------------------------
# Balance of nuisance factors
# ---------------------------------------------------------------------------------------------------------------

def level(record: dict, name: str) -> str:
    value = record[name]
    if isinstance(value, (list, tuple)):
        return "/".join(value)
    return "none" if value is None else str(value)


def balance_table(records: list[dict], factors=BALANCE_FACTORS, by=BALANCE_BY) -> dict:
    """Crosstabs of each factor against each `by` variable: out[factor][by] = {"counts": {by level: {factor level:
    n}}, "overall": {factor level: share}, "flags": [...]}. A flag marks a factor level whose share within a
    `by` level differs from its overall share by more than 0.10, for `by` levels with at least 10 records (a
    smaller group's shares are noise). A flag means the factor is confounded with that `by` variable here."""
    out: dict = {}
    for factor in factors:
        overall = Counter(level(r, factor) for r in records)
        out[factor] = {}
        for by_name in by:
            counts: dict[str, Counter] = defaultdict(Counter)
            for r in records:
                counts[level(r, by_name)][level(r, factor)] += 1
            flags = []
            for by_level, group in sorted(counts.items()):
                n = sum(group.values())
                if n < MIN_GROUP:
                    continue
                for factor_level in sorted(overall):
                    share, base = group[factor_level] / n, overall[factor_level] / len(records)
                    if abs(share - base) > FLAG_DEVIATION:
                        flags.append({"by_level": by_level, "factor_level": factor_level, "n": n,
                                      "share": round(share, 3), "overall_share": round(base, 3)})
            out[factor][by_name] = {"counts": {k: dict(v) for k, v in sorted(counts.items())},
                                    "overall": {k: round(v / len(records), 3) for k, v in sorted(overall.items())},
                                    "flags": flags}
    return out


def balance_markdown(balances: dict[str, tuple[int, dict]]) -> str:
    lines = ["# Balance of nuisance factors", "",
             f"Each cell: count (share of the regime / correct-option column). `*` marks a share more than "
             f"{FLAG_DEVIATION:.2f} from the factor level's overall share, in a column of at least {MIN_GROUP} records.", ""]
    summary = [[name, n, sum(len(per_by[b]["flags"]) for per_by in tab.values() for b in per_by)]
               for name, (n, tab) in balances.items()]
    lines += table(["dataset", "n", "flags"], summary)
    for name, (n, tab) in balances.items():
        lines += ["", f"## {name} (n = {n})"]
        for by_name in next(iter(tab.values())):
            lines += ["", f"By {by_name}", ""]
            by_levels = sorted({b for f in tab for b in tab[f][by_name]["counts"]})
            rows = []
            for factor, per_by in tab.items():
                cell = per_by[by_name]
                flagged = {(f["by_level"], f["factor_level"]) for f in cell["flags"]}
                for factor_level, overall_share in cell["overall"].items():
                    row = [f"{factor}={factor_level}", f"{overall_share:.0%}"]
                    for b in by_levels:
                        group = cell["counts"].get(b, {})
                        total = sum(group.values())
                        c = group.get(factor_level, 0)
                        row.append(f"{c} ({c / total:.0%})" + ("*" if (b, factor_level) in flagged else "") if total else "-")
                    rows.append(row)
            lines += table(["factor level", "overall", *by_levels], rows)
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------------------------
# Minimal-pair bank
# ---------------------------------------------------------------------------------------------------------------

MP_RATIO_RANGE = (1.3, 1.6)    # H = D / r (near_only) and D * r (both), r drawn per item
MP_ITEMS = 120
TWO_DECIMALS = re.compile(r"\d+\.\d[1-9] years")


def draw_horizons(rng: random.Random, far_delay, ratio_range, decimals: int) -> tuple[float, dict]:
    """A ratio r in `ratio_range` and the horizons D / r and D * r, redrawn until both are written with exactly
    two decimals, the last not 0. Otherwise D * r is often a whole number ("14 years") and D / r never is, so
    the number format would carry the regime and a direction fitted on the pairs could read it."""
    while True:
        ratio = rng.uniform(*ratio_range)
        horizons = {"short": horizon_in_years(far_delay.years / ratio, decimals),
                    "long": horizon_in_years(far_delay.years * ratio, decimals)}
        if all(TWO_DECIMALS.fullmatch(str(h)) for h in horizons.values()):
            return ratio, horizons


def build_minimal_pairs(seed: int = SEED, n_items: int = MP_ITEMS, ratio_range=MP_RATIO_RANGE, decimals: int = 2):
    """Each item has one near and one far option and is rendered at a short horizon D/r (only the near option
    pays) and a long one D*r (both pay), with r drawn per item from `ratio_range` (see `draw_horizons`), each in both option orders: 4 prompts, uid "<item>-<short|long>-
    <near|far>-first", pair_id = the item. The two prompts of an order differ only in the horizon text.

    Items are balanced by construction: every (far delay, reward multiplier) combination (6 x 5 = 30) gets
    n_items / 30 items, one per phrasing in turn, with the label style alternating within the combination, so
    far delays, multipliers and phrasings are exactly balanced and label styles are too. Per item the random draws
    are: label-style order within the combination (shuffled), ratio, near delay (uniform over those shorter than the
    short horizon, as the verification grid), near reward. The stream is seeded with "<seed>-minimal-pairs"."""
    combos = [(d, m) for d in FAR_DELAYS for m in REWARD_MULTIPLIERS]
    reps = n_items // len(combos)
    if reps % len(CONSTRAINT_PHRASINGS) or reps * len(combos) != n_items:
        raise ValueError(f"n_items must be a multiple of {len(combos) * len(CONSTRAINT_PHRASINGS)}, got {n_items}")
    rng = random.Random(f"{seed}-minimal-pairs")
    plan = []
    for far_delay, multiplier in combos:
        styles = [LABEL_STYLES[k % len(LABEL_STYLES)] for k in range(reps)]
        rng.shuffle(styles)
        plan += [(far_delay, multiplier, k % len(CONSTRAINT_PHRASINGS), styles[k]) for k in range(reps)]
    rng.shuffle(plan)

    records = []
    for i, (far_delay, multiplier, phrasing_id, labels) in enumerate(plan, start=1):
        item = f"m{i:03d}"
        ratio, horizons = draw_horizons(rng, far_delay, ratio_range, decimals)
        near_delay = rng.choice([d for d in NEAR_DELAYS if d.days < horizons["short"].days])
        near_reward = rng.choice(NEAR_REWARDS)
        for which, horizon in horizons.items():
            if (horizon.days < far_delay.days) != (which == "short"):
                raise AssertionError(f"{item}: rounding moved H={horizon} across D={far_delay}")
            for near_first in (True, False):
                records.append(make_record(
                    uid=f"{item}-{which}-{'near' if near_first else 'far'}-first", split="minimal_pair",
                    near_delay=near_delay, far_delay=far_delay, near_reward=near_reward, multiplier=multiplier,
                    near_first=near_first, labels=labels, horizon=horizon, grid_value=horizon,
                    phrasing_id=phrasing_id, pair_id=item,
                    notes={"horizon_role": which, "item_ratio": ratio}))
    require_distinct_texts(records)
    return records


def minimal_pair_counts(records) -> dict:
    return {"prompts": len(records),
            "regime": dict(Counter(r.regime for r in records)),
            "far_delay": dict(Counter(r.far.delay for r in records)),
            "reward_multiplier": dict(Counter(str(r.reward_multiplier) for r in records)),
            "phrasing_id": dict(Counter(str(r.phrasing_id) for r in records)),
            "labels": dict(Counter("/".join(r.labels) for r in records)),
            "near_first": sum(r.near_first for r in records)}


def write_minimal_pairs(out: Path, t0_bank: Path, seed: int = SEED) -> list:
    records = build_minimal_pairs(seed)
    require_distinct_texts(records + [uid_and_text(r) for r in read_jsonl(t0_bank / "prompts.jsonl")])
    write_jsonl(out / "prompts.jsonl", (record_to_dict(r) for r in records))
    manifest = {"tier": "T0", "split": "minimal_pair", "prompts_sha256": prompts_sha256(records),
                "created_utc": utc_now(), "code_commit": git_commit(),
                "config": {"seed": seed, "n_items": MP_ITEMS, "ratio_range": MP_RATIO_RANGE, "decimals": 2,
                           "distinct_from": t0_bank.name},
                "counts": minimal_pair_counts(records)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return records


def uid_and_text(row: dict):
    """What pbank's hash and distinctness check read from a record, taken from a row dict."""
    return type("Row", (), {"uid": row["uid"], "text": row["text"]})


# ---------------------------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------------------------

def write_splits(bank: Path, out: Path, seed: int = SEED) -> dict[str, list[dict]]:
    records = read_jsonl(bank / "prompts.jsonl")
    bank_sha = prompts_sha256(uid_and_text(r) for r in records)
    splits = build_splits(records, seed)
    balances = {}
    for split in ("extraction", "verification"):
        rs = [r for r in records if r["split"] == split]
        balances[f"{bank.name} {split}"] = (len(rs), balance_table(rs))
    for name, rows in splits.items():
        write_jsonl(out / f"{name}.jsonl", rows)
        (out / f"{name}.manifest.json").write_text(json.dumps(split_manifest(name, rows, bank_sha, seed), indent=2) + "\n")
        if name == "debatable":
            continue
        for role in sorted({r["role"] for r in rows}):
            members = rows_of_split(records, rows, role)
            balances[f"{name}/{role}"] = (len(members), balance_table(members))
    (out / "balance.json").write_text(json.dumps({k: v[1] for k, v in balances.items()}, indent=1) + "\n")
    (out / "balance.md").write_text(balance_markdown(balances))
    return splits


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p_splits = sub.add_parser("splits", help="write the index files and balance tables")
    p_splits.add_argument("--bank", type=Path, required=True)
    p_splits.add_argument("--out", type=Path, help="default: <bank>/splits")
    p_splits.add_argument("--seed", type=int, default=SEED)
    p_pairs = sub.add_parser("minimal-pairs", help="write the minimal-pair bank")
    p_pairs.add_argument("--out", type=Path, required=True)
    p_pairs.add_argument("--t0-bank", type=Path, default=Path(__file__).resolve().parents[1] / "prompt_bank/data/t0_s0")
    p_pairs.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    if args.command == "splits":
        splits = write_splits(args.bank, args.out or args.bank / "splits", args.seed)
        for name, rows in splits.items():
            print(f"{name:<24} {len(rows):>4} rows  {dict(Counter(r['role'] for r in rows))}")
    else:
        records = write_minimal_pairs(args.out, args.t0_bank, args.seed)
        print(f"wrote {len(records)} prompts to {args.out}")


if __name__ == "__main__":
    main()
