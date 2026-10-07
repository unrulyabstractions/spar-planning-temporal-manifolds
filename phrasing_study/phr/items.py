"""Single-turn items: content x variant for the `choice` and `state` protocols, all text from variants.yaml.

Content cell = option configuration x domain x horizon level x option order. Every variant is rendered on every cell
where it is defined, so a variant and its reference differ only in the variant's own change (paired design).
Prompts are rendered from the layout templates in variants.yaml (`choice.layouts`).
Columns: item_id, protocol, family, variant, ref_variant, layout, cell, config, domain, level, h_years, h_text,
short_first, label_short, label_long, meaning_shift, cue, cue_direction, implicit_id, with_number, determinacy, text.
"""

from __future__ import annotations

import itertools
import math
import re
from pathlib import Path

import pandas as pd
import yaml

from .durations import has_duration, parse_years

HERE = Path(__file__).resolve().parent.parent
VARIANTS_YAML = HERE / "variants.yaml"
CANONICAL_LAYOUT = "formatted"


def load_config(path: Path = VARIANTS_YAML) -> dict:
    return yaml.safe_load(Path(path).read_text())


def cue_table(cfg: dict) -> dict[str, dict]:
    """cue id (<direction>_<n>) -> {text, direction}."""
    return {f"{d}_{i}": {"text": t, "direction": d}
            for d, texts in cfg["choice"]["cues"].items() for i, t in enumerate(texts, start=1)}


# ---- variants -----------------------------------------------------------------------------------------------

def choice_variants(cfg: dict) -> list[dict]:
    """Canonical + every family's variants + layout variants + cues + the layout cross, as flat dicts
    (id, family, ref, plus override fields)."""
    ch = cfg["choice"]
    out = [{"id": "canonical", "family": "canonical", "ref": None}]
    base = {}
    for fam, entries in ch["families"].items():
        if fam == "unit":
            vs = [{"id": f, "family": "unit", "ref": "canonical", "horizon_form": f} for f in entries["forms"]]
        else:
            vs = [{"family": fam, "ref": "canonical", **e} for e in entries]
        out += vs
        base.update({v["id"]: v for v in vs})
    for lid in ch["layouts"]:
        if lid != CANONICAL_LAYOUT:
            out.append({"id": lid, "family": "layout", "ref": "canonical", "layout": lid})
    cues = []
    for cid, c in cue_table(cfg).items():
        cues.append({"id": cid, "family": "cue", "ref": "canonical", "extra": c["text"], "cue": cid,
                     "cue_direction": c["direction"]})
    out += cues
    base.update({v["id"]: v for v in cues})
    for lid, vid in itertools.product(ch["layouts"], ch["layout_cross"]):
        if lid != CANONICAL_LAYOUT:
            v = base[vid]
            out.append({**v, "id": f"{lid}+{vid}", "family": "layout_cross", "ref": lid, "layout": lid,
                        "base_variant": vid})
    return out


def _fields(cfg: dict, v: dict) -> dict:
    """Text fields of the variant's layout with the variant's overrides applied (unset layout fields fall back to the
    canonical layout's)."""
    lays = cfg["choice"]["layouts"]
    lay = {**lays[CANONICAL_LAYOUT], **lays[v.get("layout", CANONICAL_LAYOUT)]}
    return {"layout": v.get("layout", CANONICAL_LAYOUT), "template": lay["template"],
            "constraint": v.get("constraint", lay["constraint"]), "objective": v.get("objective", lay["objective"]),
            "action": v.get("action", lay["action"]), "labels": tuple(v.get("labels", ("a)", "b)"))),
            "extra": v.get("extra"), "option_unit": v.get("option_unit"),
            "implicit_frame": lays[v.get("layout", CANONICAL_LAYOUT)].get("implicit_frame"),
            "twin_frame": lays[v.get("layout", CANONICAL_LAYOUT)].get("twin_frame")}


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


_FIELD = re.compile(r"\{(\w+)\}")
_OPTIONAL = re.compile(r"\[([^\[\]]*)\]")


def fill_template(template: str, values: dict) -> str:
    """Fill {name} placeholders; a [bracketed] part is dropped if any placeholder inside it is empty/None; lines left
    empty are dropped. Literal square brackets are therefore not allowed in templates."""
    def opt(m):
        part = m.group(1)
        return "" if any(not values.get(k) for k in _FIELD.findall(part)) else part
    text = _OPTIONAL.sub(opt, template)
    missing = [k for k in _FIELD.findall(text) if values.get(k) is None]
    if missing:
        raise KeyError(f"template placeholders without a value: {missing}")
    text = _FIELD.sub(lambda m: str(values[m.group(1)]), text)
    return "\n".join(line.rstrip() for line in text.split("\n") if line.strip())


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
    values = {**dom, "la": la, "lb": lb, "first": first, "second": second, "objective": f["objective"],
              "constraint": constraint, "extra": f["extra"], "action": action, "format": fmt}
    return fill_template(f["template"], values)


# ---- item tables ----------------------------------------------------------------------------------------------

def _row(protocol, fam, vid, ref, cell, conf, domain, level, h_years, h_text, short_first, f, text, **kw):
    la, lb = f["labels"]
    return dict(protocol=protocol, family=fam, variant=vid, ref_variant=ref, layout=f["layout"], cell=cell,
                config=conf["id"], domain=domain, level=level, h_years=h_years, h_text=h_text, short_first=short_first,
                label_short=la if short_first else lb, label_long=lb if short_first else la,
                meaning_shift=bool(kw.pop("meaning_shift", False)), cue=kw.pop("cue", None),
                cue_direction=kw.pop("cue_direction", None), base_variant=kw.pop("base_variant", None),
                implicit_id=kw.pop("implicit_id", None), with_number=kw.pop("with_number", None),
                determinacy=kw.pop("determinacy", None), text=text)


def _implicit_rows(cfg, protocol, f, configs, orders, suffix=""):
    """Implicit items and their twins in the layout of `f` (investment domain)."""
    rows = []
    for it, conf, sf in itertools.product(cfg["choice"]["implicit"], configs, orders):
        cell = f"{conf['id']}|investment|{it['id']}|{'S' if sf else 'L'}"
        yrs = parse_years(it["twin"])
        for kind, cons, h_text in (("implicit", f["implicit_frame"].format(when=it["when"]), it["when"]),
                                   ("twin", f["twin_frame"].format(h=it["twin"]), it["twin"])):
            rows.append(_row(protocol, kind, f"{kind}:{it['id']}{suffix}",
                             f"twin:{it['id']}{suffix}" if kind == "implicit" else None, cell, conf, "investment",
                             it["id"], yrs, h_text, sf, f, render(cfg, f, "investment", conf, cons, sf, protocol),
                             implicit_id=it["id"], with_number=it["with_number"], determinacy=it["determinacy"]))
    return rows


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
                             cue=v.get("cue"), cue_direction=v.get("cue_direction"), base_variant=v.get("base_variant")))
    # no horizon: the constraint is dropped, with and without cues
    cues = cue_table(cfg)
    for cue in ch["no_horizon_cues"]:
        cue = None if cue == "none" else cue
        f = _fields(cfg, {"extra": cues[cue]["text"]} if cue else {})
        for conf, domain, sf in itertools.product(ch["configs"], ch["domains"], (True, False)):
            cell = f"{conf['id']}|{domain}|none|{'S' if sf else 'L'}"
            rows.append(_row("choice", "no_horizon", f"nohorizon_{cue or 'none'}", None if cue is None else "nohorizon_none",
                             cell, conf, domain, None, math.nan, None, sf, f, render(cfg, f, domain, conf, None, sf),
                             cue=cue, cue_direction=cues[cue]["direction"] if cue else None))
    # implicit horizons and their explicit twins, in every layout that defines the frames
    for lid, lay in ch["layouts"].items():
        if lay.get("implicit_frame") and lay.get("twin_frame"):
            rows += _implicit_rows(cfg, "choice", _fields(cfg, {"layout": lid}), ch["configs"], (True, False),
                                   "" if lid == CANONICAL_LAYOUT else f"@{lid}")
    df = pd.DataFrame(rows)
    df.insert(0, "item_id", [f"ch{i:05d}" for i in range(len(df))])
    return df


def build_state(cfg: dict) -> pd.DataFrame:
    """Every distinct horizon text (canonical + forms) on one cell per domain, every implicit item + twin
    (investment), and the no-horizon prompt. Canonical layout; one cell = first config, short option first."""
    ch = cfg["choice"]
    conf = ch["configs"][0]
    f = _fields(cfg, {})
    rows = []
    for h, domain in itertools.product(cfg["horizons"], ch["domains"]):
        for vid, h_text in [("canonical", h["canonical"]), *h["forms"].items()]:
            text = render(cfg, f, domain, conf, f["constraint"].format(h=h_text), True, "state")
            rows.append(_row("state", "unit" if vid != "canonical" else "canonical", vid, "canonical",
                             f"{conf['id']}|{domain}|{h['canonical']}|S", conf, domain, h["canonical"],
                             parse_years(h["canonical"]), h_text, True, f, text))
    rows += _implicit_rows(cfg, "state", f, [conf], (True,))
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
    ch, mt = cfg["choice"], cfg["multiturn"]
    # YAML reads bare null/no/yes/on/off keys as None/False/True: every name must be a string
    names = (list(ch["families"]) + list(ch["layouts"]) + list(ch["cues"])
             + [e.get("id") for fam, es in ch["families"].items() if fam != "unit" for e in es])
    bad = [n for n in names if not isinstance(n, str)]
    errs += [f"name {n!r} is not a string (quote it in variants.yaml)" for n in bad]
    if bad:
        return errs
    forms = set(ch["families"]["unit"]["forms"])
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
    if CANONICAL_LAYOUT not in ch["layouts"]:
        return errs + [f"layout {CANONICAL_LAYOUT!r} (the canonical one) is missing"]
    for k in ("constraint", "objective", "action", "template", "implicit_frame", "twin_frame"):
        if k not in ch["layouts"][CANONICAL_LAYOUT]:
            errs.append(f"canonical layout lacks {k!r}")
    for lid, lay in ch["layouts"].items():
        if "{format}" not in lay.get("template", ""):
            errs.append(f"layout {lid}: template has no {{format}} (the readout needs the answer format)")
        if "{h}" not in lay.get("constraint", "{h}"):
            errs.append(f"layout {lid}: constraint has no {{h}}")
    cue_ids = set(cue_table(cfg))
    try:
        vs = choice_variants(cfg)
    except KeyError as e:
        return errs + [f"layout_cross refers to an unknown variant: {e}"]
    ids = [v["id"] for v in vs]
    errs += [f"duplicate variant id {i!r}" for i in sorted({i for i in ids if ids.count(i) > 1})]
    for v in vs:
        if v.get("ref") and v["ref"] not in ids:
            errs.append(f"variant {v['id']}: ref {v['ref']!r} does not exist")
        if v.get("horizon_form") and v["horizon_form"] not in forms:
            errs.append(f"variant {v['id']}: horizon_form {v['horizon_form']!r} not a unit form")
        if "constraint" in v and "{h}" not in v["constraint"]:
            errs.append(f"variant {v['id']}: constraint has no {{h}}")
        if v.get("layout", CANONICAL_LAYOUT) not in ch["layouts"]:
            errs.append(f"variant {v['id']}: unknown layout {v['layout']!r}")
    for it in ch["implicit"]:
        if parse_years(it["twin"]) is None:
            errs.append(f"implicit {it['id']}: cannot parse twin {it['twin']!r}")
        if not it["with_number"] and (has_duration(it["when"]) or any(c.isdigit() for c in it["when"])):
            errs.append(f"implicit {it['id']}: with_number is false but the text has a number/duration")
        if it.get("determinacy") not in ("concrete", "vague"):
            errs.append(f"implicit {it['id']}: determinacy must be concrete or vague")
    errs += [f"no_horizon cue {c!r} unknown" for c in ch["no_horizon_cues"] if c != "none" and c not in cue_ids]
    known = {i["id"] for i in ch["implicit"]}
    errs += [f"multiturn implicit item {i!r} unknown" for i in mt["implicit_items"] if i not in known]
    errs += [f"multiturn horizon {h!r} is not a horizon level" for h in mt["horizons"]
             if h not in {x["canonical"] for x in cfg["horizons"]}]
    errs += [f"multiturn cue {c!r} unknown" for c in mt["first_turn_cues"] + mt["later_cues"] if c not in cue_ids]
    return errs
