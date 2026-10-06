"""Single-turn items: content x variant for the `choice` and `state` protocols, all text from variants.yaml.

Content cell = option configuration x domain x horizon level x option order. Every variant is rendered on every cell
where it is defined, so a variant and its reference differ only in the variant's own change (paired design).
Columns: item_id, protocol, family, variant, ref_variant, cell, config, domain, level, h_years, h_text, short_first,
label_short, label_long, meaning_shift, cue, implicit_id, with_number, text.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import pandas as pd
import yaml

from .durations import has_duration, parse_years

HERE = Path(__file__).resolve().parent.parent
VARIANTS_YAML = HERE / "variants.yaml"


def load_config(path: Path = VARIANTS_YAML) -> dict:
    return yaml.safe_load(Path(path).read_text())


# ---- variants -----------------------------------------------------------------------------------------------

def choice_variants(cfg: dict) -> list[dict]:
    """Canonical + every family's variants as flat dicts: id, family, ref, plus override fields."""
    fams = cfg["choice"]["families"]
    out = [{"id": "canonical", "family": "canonical", "ref": None}]
    for fam, entries in fams.items():
        if fam == "unit":
            out += [{"id": f, "family": "unit", "ref": "canonical", "horizon_form": f} for f in entries["forms"]]
            continue
        for e in entries:
            out.append({"family": fam, "ref": "canonical", **e})
    return out


def _fields(cfg: dict, v: dict) -> dict:
    """Canonical text fields with the variant's overrides applied."""
    c = cfg["choice"]["canonical"]
    return {"constraint": v.get("constraint", c["constraint"]), "objective": v.get("objective", c["objective"]),
            "action": v.get("action", c["action"]), "labels": tuple(v.get("labels", c["labels"])),
            "extra": v.get("extra"), "layout": v.get("layout", "formatted"),
            "constraint_first": bool(v.get("constraint_first", False)), "option_unit": v.get("option_unit")}


# ---- rendering ----------------------------------------------------------------------------------------------

def _delay_text(text: str, option_unit: str | None) -> str:
    if option_unit is None:
        return text
    if option_unit != "months":
        raise ValueError(f"option_unit {option_unit!r} not supported (months)")
    n = round(parse_years(text) * 12)
    return f"{n} month" + ("s" if n != 1 else "")


def option_lines(dom: dict, conf: dict, option_unit: str | None) -> tuple[str, str]:
    short = f"{conf['reward']:,} {dom['reward_unit']} in {_delay_text(conf['short'], option_unit)}."
    long = f"{conf['reward'] * conf['mult']:,} {dom['reward_unit']} in {_delay_text(conf['long'], option_unit)}."
    return short, long


def render(cfg: dict, f: dict, domain: str, conf: dict, constraint: str | None, short_first: bool,
           protocol: str = "choice") -> str:
    """Prompt text. `constraint`: the full constraint sentence (horizon already filled in) or None (no horizon)."""
    dom = cfg["choice"]["domains"][domain]
    short, long = option_lines(dom, conf, f["option_unit"])
    first, second = (short, long) if short_first else (long, short)
    la, lb = f["labels"]
    if protocol == "state":
        action, fmt = cfg["state"]["action"], cfg["state"]["format"]
    else:
        action, fmt = f["action"], f"I choose: <{la} or {lb}>. My reasoning: <1-3 sentences>"
    if f["layout"] == "prose":
        parts = [f"You are {dom['role']}. {dom['situation']} You are tasked to {dom['task']}. Two options are available:",
                 f"{la} {first}", f"{lb} {second}",
                 " ".join(x for x in (f["objective"], constraint, f["extra"]) if x),
                 f"{action} Answer in this format: {fmt}" if protocol == "choice" else f"{action} {fmt}"]
        return "\n".join(parts)
    cons = [f"CONSTRAINT: {constraint}"] if constraint else []
    lines = [f"SITUATION: {dom['situation']}", f"TASK: You, {dom['role']}, are tasked to {dom['task']}:",
             *(cons if f["constraint_first"] else []), f"{la} {first}", f"{lb} {second}",
             f"OBJECTIVE: {f['objective']}", *([] if f["constraint_first"] else cons),
             *([f["extra"]] if f["extra"] else []), f"ACTION: {action}", f"FORMAT: {fmt}"]
    return "\n".join(lines)


# ---- item tables ----------------------------------------------------------------------------------------------

def _row(protocol, fam, vid, ref, cell, conf, domain, level, h_years, h_text, short_first, f, text, **kw):
    la, lb = f["labels"]
    return dict(protocol=protocol, family=fam, variant=vid, ref_variant=ref, cell=cell, config=conf["id"],
                domain=domain, level=level, h_years=h_years, h_text=h_text, short_first=short_first,
                label_short=la if short_first else lb, label_long=lb if short_first else la,
                meaning_shift=bool(kw.pop("meaning_shift", False)), cue=kw.pop("cue", None),
                implicit_id=kw.pop("implicit_id", None), with_number=kw.pop("with_number", None), text=text)


def build_choice(cfg: dict) -> pd.DataFrame:
    ch = cfg["choice"]
    levels = {h["canonical"]: h for h in cfg["horizons"]}
    rows = []
    cells = list(itertools.product(ch["configs"], ch["domains"], levels, (True, False)))
    for v in choice_variants(cfg):
        f = _fields(cfg, v)
        for conf, domain, level, sf in cells:
            form = v.get("horizon_form")
            h_text = levels[level]["forms"].get(form) if form else level
            if h_text is None:
                continue                                  # form not defined at this level
            cell = f"{conf['id']}|{domain}|{level}|{'S' if sf else 'L'}"
            text = render(cfg, f, domain, conf, f["constraint"].format(h=h_text), sf)
            rows.append(_row("choice", v["family"], v["id"], v.get("ref"), cell, conf, domain, level,
                             parse_years(level), h_text, sf, f, text, meaning_shift=v.get("meaning_shift"),
                             cue=v["id"] if v["family"] == "cue" else None))
    # no horizon: canonical template without the CONSTRAINT line, with and without cues
    cues = {v["id"]: v for v in ch["families"]["cue"]}
    for cue in ch["no_horizon_cues"]:
        cue = None if cue == "none" else cue
        v = {"extra": cues[cue]["extra"]} if cue else {}
        f = _fields(cfg, v)
        for conf, domain, sf in itertools.product(ch["configs"], ch["domains"], (True, False)):
            cell = f"{conf['id']}|{domain}|none|{'S' if sf else 'L'}"
            rows.append(_row("choice", "no_horizon", f"nohorizon_{cue or 'none'}", None if cue is None else "nohorizon_none",
                             cell, conf, domain, None, math.nan, None, sf, f, render(cfg, f, domain, conf, None, sf), cue=cue))
    # implicit horizons and their explicit twins (investment domain only)
    f = _fields(cfg, {})
    for it, conf, sf in itertools.product(ch["implicit"], ch["configs"], (True, False)):
        cell = f"{conf['id']}|investment|{it['id']}|{'S' if sf else 'L'}"
        yrs = parse_years(it["twin"])
        for kind, cons, h_text in (("implicit", ch["implicit_frame"].format(when=it["when"]), it["when"]),
                                   ("twin", ch["twin_frame"].format(h=it["twin"]), it["twin"])):
            rows.append(_row("choice", kind, f"{kind}:{it['id']}", f"twin:{it['id']}" if kind == "implicit" else None,
                             cell, conf, "investment", it["id"], yrs, h_text, sf, f,
                             render(cfg, f, "investment", conf, cons, sf), implicit_id=it["id"],
                             with_number=it["with_number"]))
    df = pd.DataFrame(rows)
    df.insert(0, "item_id", [f"ch{i:05d}" for i in range(len(df))])
    return df


def build_state(cfg: dict) -> pd.DataFrame:
    """Every distinct horizon text (canonical + forms) on one cell per domain, every implicit item + twin
    (investment), and the no-horizon prompt. One cell = first config, short option first."""
    ch = cfg["choice"]
    conf = ch["configs"][0]
    f = _fields(cfg, {})
    rows = []
    for h, domain in itertools.product(cfg["horizons"], ch["domains"]):
        for vid, h_text in [("canonical", h["canonical"]), *h["forms"].items()]:
            text = render(cfg, f, domain, conf, ch["canonical"]["constraint"].format(h=h_text), True, "state")
            rows.append(_row("state", "unit" if vid != "canonical" else "canonical", vid, "canonical",
                             f"{conf['id']}|{domain}|{h['canonical']}|S", conf, domain, h["canonical"],
                             parse_years(h["canonical"]), h_text, True, f, text))
    for it in ch["implicit"]:
        cell = f"{conf['id']}|investment|{it['id']}|S"
        for kind, cons, h_text in (("implicit", ch["implicit_frame"].format(when=it["when"]), it["when"]),
                                   ("twin", ch["twin_frame"].format(h=it["twin"]), it["twin"])):
            rows.append(_row("state", kind, f"{kind}:{it['id']}", f"twin:{it['id']}" if kind == "implicit" else None,
                             cell, conf, "investment", it["id"], parse_years(it["twin"]), h_text, True, f,
                             render(cfg, f, "investment", conf, cons, True, "state"), implicit_id=it["id"],
                             with_number=it["with_number"]))
    for domain in ch["domains"]:
        rows.append(_row("state", "no_horizon", "nohorizon_none", None, f"{conf['id']}|{domain}|none|S", conf, domain,
                         None, math.nan, None, True, f, render(cfg, f, domain, conf, None, True, "state")))
    df = pd.DataFrame(rows)
    df.insert(0, "item_id", [f"st{i:04d}" for i in range(len(df))])
    return df


# ---- validation -----------------------------------------------------------------------------------------------

def check_config(cfg: dict) -> list[str]:
    """Problems in variants.yaml (empty list = OK)."""
    errs = []
    forms = set(cfg["choice"]["families"]["unit"]["forms"])
    for h in cfg["horizons"]:
        y = parse_years(h["canonical"])
        if y is None:
            errs.append(f"cannot parse canonical horizon {h['canonical']!r}")
            continue
        for k, t in h["forms"].items():
            if k not in forms:
                errs.append(f"{h['canonical']}: form {k!r} is not listed in choice.families.unit.forms")
            ty = parse_years(t)
            if ty is None or abs(ty / y - 1) > 0.10:
                errs.append(f"{h['canonical']}: form {k}={t!r} parses to {ty}, not within 10% of {y:.4g} years")
    # YAML reads bare null/no/yes/on/off keys as None/False/True: every family, variant and cue name must be a string
    names = list(cfg["choice"]["families"]) + list(cfg["multiturn"]["cues"]) + [v.get("id") for v in choice_variants(cfg)]
    errs += [f"name {n!r} is not a string (quote it in variants.yaml)" for n in names if not isinstance(n, str)]
    ids = [v["id"] for v in choice_variants(cfg)]
    errs += [f"duplicate variant id {i!r}" for i in sorted({i for i in ids if ids.count(i) > 1})]
    for v in choice_variants(cfg):
        if v.get("ref") and v["ref"] not in ids:
            errs.append(f"variant {v['id']}: ref {v['ref']!r} does not exist")
        if v.get("horizon_form") and v["horizon_form"] not in forms:
            errs.append(f"variant {v['id']}: horizon_form {v['horizon_form']!r} not a unit form")
        for key in ("constraint",):
            if key in v and "{h}" not in v[key]:
                errs.append(f"variant {v['id']}: {key} has no {{h}}")
    for it in cfg["choice"]["implicit"]:
        if parse_years(it["twin"]) is None:
            errs.append(f"implicit {it['id']}: cannot parse twin {it['twin']!r}")
        if not it["with_number"] and (has_duration(it["when"]) or any(c.isdigit() for c in it["when"])):
            errs.append(f"implicit {it['id']}: with_number is false but the text has a number/duration")
    mt = cfg["multiturn"]
    known = {i["id"] for i in cfg["choice"]["implicit"]}
    errs += [f"multiturn implicit item {i!r} unknown" for i in mt["implicit_items"] if i not in known]
    errs += [f"multiturn horizon {h!r} is not a horizon level" for h in mt["horizons"]
             if h not in {x["canonical"] for x in cfg["horizons"]}]
    errs += [f"no_horizon cue {c!r} unknown" for c in cfg["choice"]["no_horizon_cues"]
             if c != "none" and c not in {v["id"] for v in cfg["choice"]["families"]["cue"]}]
    errs += [f"multiturn cue {c!r} unknown" for c in mt["first_turn_cues"] + mt["later_cues"] if c not in mt["cues"]]
    return errs
