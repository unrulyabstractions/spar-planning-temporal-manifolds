"""Behavioural analysis of a run. Everything is a paired difference against a reference on the same content cell,
with a bootstrap CI that resamples scenarios (clusters), and where possible converted to the common currency:
a shift in log10 effective horizon (multiplier 10^shift: "acts like a horizon x times the stated one").

choice     Δ P(short) per variant vs its reference; null-rewording floor; effective-horizon shift from a fixed-effects
           fit of the model's log-odds on log10 horizon; contrasts (cue pairs, order, unit match).
state      stated horizon vs the true one (unit forms, implicit vs twin).
multiturn  plan end (largest parsed step horizon) vs the reference conversation; step-cue effects on later steps.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

N_BOOT = 2000
MIN_EFFECT = 0.05           # |mean Δ P(short)| below this never counts as "matters"
CLIP = 10.0                 # log-odds clip for the effective-horizon fit
MIN_SLOPE = 0.5             # |log-odds per decade of horizon| needed to read a shift as a horizon change


def boot_ci(df: pd.DataFrame, value: str, cluster: str, n: int = N_BOOT, seed: int = 0) -> tuple[float, float, float]:
    """Mean of `value` and a 95% percentile CI resampling clusters (each cluster's cells kept together)."""
    d = df[[cluster, value]].dropna()
    if d.empty:
        return math.nan, math.nan, math.nan
    g = d.groupby(cluster)[value].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(n, len(g)))
    boots = s[idx].sum(1) / c[idx].sum(1)
    return float(s.sum() / c.sum()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# ---- choice ---------------------------------------------------------------------------------------------------

def paired(choice: pd.DataFrame) -> pd.DataFrame:
    """One row per (variant item, its reference item on the same cell)."""
    ref = choice[["variant", "cell", "p_short", "logit_diff"]].rename(
        columns={"variant": "ref_variant", "p_short": "p_ref", "logit_diff": "ld_ref"})
    d = choice[choice.ref_variant.notna()].merge(ref, on=["ref_variant", "cell"], how="inner")
    d["delta"] = d.p_short - d.p_ref
    d["scenario"] = d.config + "|" + d.domain
    return d


def effective_shift(choice: pd.DataFrame, variant: str, ref: str, n_boot: int = 500) -> dict:
    """Fit clip(log-odds short) = a_(config,domain,order) + b*log10(h) + g*[variant] on the variant's and the reference's
    items at levels where both exist. Shift = g/b in log10 years (negative = acts like a shorter horizon)."""
    d = choice[choice.variant.isin([variant, ref]) & choice.h_years.notna()].copy()
    lv = set(d[d.variant == variant].level) & set(d[d.variant == ref].level)
    d = d[d.level.isin(lv)]
    if len(lv) < 3:
        return {"shift_log10": math.nan, "lo": math.nan, "hi": math.nan, "slope": math.nan, "n_levels": len(lv)}
    d["base"] = d.config + "|" + d.domain + "|" + d.short_first.astype(str)
    d["scenario"] = d.config + "|" + d.domain

    def fit(x: pd.DataFrame) -> tuple[float, float]:
        bases = sorted(x.base.unique())
        X = np.zeros((len(x), len(bases) + 2))
        X[np.arange(len(x)), x.base.map({b: i for i, b in enumerate(bases)}).to_numpy()] = 1
        X[:, -2] = np.log10(x.h_years.to_numpy())
        X[:, -1] = (x.variant == variant).to_numpy()
        coef = np.linalg.lstsq(X, np.clip(x.logit_diff.to_numpy(), -CLIP, CLIP), rcond=None)[0]
        return coef[-1] / coef[-2], coef[-2]

    shift, slope = fit(d)
    rng = np.random.default_rng(0)
    scen = d.scenario.unique()
    groups = {s: g for s, g in d.groupby("scenario")}
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(scen, size=len(scen))
        x = pd.concat([groups[s].assign(base=groups[s].base + f"#{i}") for i, s in enumerate(pick)])
        boots.append(fit(x)[0])
    return {"shift_log10": shift, "lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5)),
            "slope": slope, "n_levels": len(lv)}


def choice_tables(choice: pd.DataFrame) -> dict[str, pd.DataFrame]:
    d = paired(choice)
    rows = []
    for (fam, var, ref), g in d.groupby(["family", "variant", "ref_variant"], sort=False):
        m, lo, hi = boot_ci(g, "delta", "scenario")
        rows.append(dict(family=fam, variant=var, ref=ref, n=len(g), mean_delta=m, lo=lo, hi=hi,
                         mean_abs_delta=g.delta.abs().mean(), meaning_shift=bool(g.meaning_shift.any())))
    per = pd.DataFrame(rows)
    nulls = per[per.family == "nulls"]
    floor = float(nulls.mean_delta.abs().max()) if len(nulls) else 0.0
    per["above_floor"] = per.mean_delta.abs() > floor
    per["matters"] = per.above_floor & ((per.lo > 0) | (per.hi < 0)) & (per.mean_delta.abs() >= MIN_EFFECT)
    per.loc[per.family == "nulls", "matters"] = False

    # effective-horizon shifts for variants defined on >= 3 horizon levels
    shifts = []
    for r in per.itertuples():
        if r.family in ("implicit", "no_horizon"):
            continue
        s = effective_shift(choice, r.variant, r.ref)
        if not math.isnan(s["shift_log10"]):
            shifts.append(dict(variant=r.variant, ref=r.ref, **s, multiplier=10 ** s["shift_log10"]))
    shifts = pd.DataFrame(shifts)
    if len(shifts):   # a shift is only meaningful where the curve actually falls with the horizon on those levels
        shifts["reliable"] = (shifts.slope <= -MIN_SLOPE) & (shifts.n_levels >= 4)

    # validity gate: canonical curve
    can = choice[choice.variant == "canonical"]
    curve = can.groupby(["level", "short_first"], sort=False).p_short.mean().unstack()
    curve["mean"] = can.groupby("level", sort=False).p_short.mean()
    curve.insert(0, "years", can.groupby("level", sort=False).h_years.first())
    rho = can.groupby("cell").first()
    from scipy.stats import spearmanr
    gate = pd.DataFrame([dict(spearman_logh_vs_p=spearmanr(np.log10(can.h_years), can.p_short).statistic,
                              p_short_lowest=curve["mean"].iloc[0], p_short_highest=curve["mean"].iloc[-1],
                              order_effect=(can[can.short_first].p_short.mean() - can[~can.short_first].p_short.mean()),
                              null_floor=floor, n_rho_cells=len(rho))])

    # contrasts between two variants on common cells
    def contrast(a: str, b: str, name: str) -> dict:
        x = choice[choice.variant == a][["cell", "config", "domain", "p_short"]].merge(
            choice[choice.variant == b][["cell", "p_short"]], on="cell", suffixes=("_a", "_b"))
        x["diff"], x["scenario"] = x.p_short_a - x.p_short_b, x.config + "|" + x.domain
        m, lo, hi = boot_ci(x, "diff", "scenario")
        return dict(contrast=name, a=a, b=b, n=len(x), mean_diff=m, lo=lo, hi=hi)

    pairs = [("lean_sooner", "lean_later", "agreement cue (sooner - later)"),
             ("hurried", "relaxed", "pressure cue (hurried - relaxed)"),
             ("nohorizon_lean_sooner", "nohorizon_lean_later", "agreement cue, no horizon"),
             ("nohorizon_hurried", "nohorizon_relaxed", "pressure cue, no horizon")]
    con = [contrast(a, b, n) for a, b, n in pairs if {a, b} <= set(choice.variant)]
    # option order inside every variant: short first - short second, same everything else
    o = choice.assign(base=choice.cell.str.rsplit("|", n=1).str[0])
    o = o.pivot_table(index=["variant", "base", "config", "domain"], columns="short_first", values="p_short").dropna().reset_index()
    if {True, False} <= set(o.columns):
        o["diff"], o["scenario"] = o[True] - o[False], o.config + "|" + o.domain
        m, lo, hi = boot_ci(o, "diff", "scenario")
        con.append(dict(contrast="option order (short first - short second), all variants", a="", b="", n=len(o),
                        mean_diff=m, lo=lo, hi=hi))
    # unit match: does expressing the horizon in months matter more when the options are in months?
    if {"unit_months", "opt_months", "opt_months_h_months"} <= set(choice.variant):
        x = choice.pivot_table(index="cell", columns="variant", values="p_short")
        x = x[["canonical", "unit_months", "opt_months", "opt_months_h_months"]].dropna()
        x["diff"] = (x.opt_months_h_months - x.opt_months) - (x.unit_months - x.canonical)
        x["scenario"] = [c.rsplit("|", 2)[0] for c in x.index]
        m, lo, hi = boot_ci(x.reset_index(), "diff", "scenario")
        con.append(dict(contrast="unit-match interaction (h months: with month options - with mixed options)",
                        a="", b="", n=len(x), mean_diff=m, lo=lo, hi=hi))
    return {"choice_variants": per, "choice_shifts": shifts, "choice_curve": curve.reset_index(),
            "choice_gate": gate, "choice_contrasts": pd.DataFrame(con)}


# ---- state ----------------------------------------------------------------------------------------------------

def state_table(state: pd.DataFrame) -> pd.DataFrame:
    s = state.copy()
    s["log_err"] = np.log10(s.stated_years / s.h_years)
    s["within_2x"] = s.log_err.abs() <= math.log10(2)
    s["exact_10pct"] = s.log_err.abs() <= math.log10(1.1)
    grp = s.assign(kind=np.where(s.family.isin(["implicit", "twin"]),
                                 s.family + np.where(s.with_number.fillna(False).astype(bool), " (with number)", " (no number)"),
                                 s.family))
    return grp.groupby("kind").agg(n=("item_id", "size"), parsed=("stated_years", lambda x: x.notna().mean()),
                                   within_2x=("within_2x", "mean"), within_10pct=("exact_10pct", "mean"),
                                   median_abs_log_err=("log_err", lambda x: x.abs().median())).reset_index()


# ---- multiturn ------------------------------------------------------------------------------------------------

def conversation_table(mt: pd.DataFrame, cue_step: int) -> pd.DataFrame:
    """One row per conversation: plan end (max parsed step horizon), last-step horizon, mean log10 of steps >= cue_step."""
    st = mt[mt.kind == "step"].copy()
    st["log_h"] = np.log10(st.h_step_years)
    st["step"] = st.turn - 1
    agg = st.groupby("conv_id").agg(n_parsed=("h_step_years", lambda x: x.notna().sum()),
                                     plan_end=("h_step_years", "max"))
    late = st[st.step >= cue_step].groupby("conv_id").log_h.mean().rename("log_late")
    conv = mt.drop_duplicates("conv_id").set_index("conv_id")[
        ["scenario", "condition", "variant", "level", "h_years", "cue", "cue_where", "greedy", "sample", "parent"]]
    out = conv.join(agg).join(late)
    out["log_end"] = np.log10(out.plan_end)
    return out.reset_index()


def multiturn_tables(mt: pd.DataFrame, cue_step: int) -> dict[str, pd.DataFrame]:
    c = conversation_table(mt, cue_step)
    rows = []

    def add(name, x, value):
        m, lo, hi = boot_ci(x, value, "scenario")
        rows.append(dict(comparison=name, n=int(x[value].notna().sum()), mean_log10=m, lo=lo, hi=hi,
                         multiplier=10 ** m if not math.isnan(m) else math.nan))

    base = c[(c.condition == "base") & c.cue.isna()]
    key = ["scenario", "level"]
    # unit forms vs canonical, per form
    for form, g in c[c.condition == "unit"].groupby("variant"):
        x = g.merge(base[key + ["log_end"]], on=key, suffixes=("", "_ref"))
        x["d"] = x.log_end - x.log_end_ref
        add(f"unit form {form} vs canonical (plan end)", x, "d")
    # implicit vs twin
    imp, twin = c[c.condition == "implicit"], c[c.condition == "twin"]
    x = imp.merge(twin[key + ["log_end"]], on=key, suffixes=("", "_ref"))
    x["d"] = x.log_end - x.log_end_ref
    add("implicit vs explicit twin (plan end)", x, "d")
    # first-turn cues vs the same conversation without cue
    for cond in ("base", "free"):
        ref = c[(c.condition == cond) & c.cue.isna()]
        k2 = key if cond == "base" else ["scenario", "sample"]
        for cue, g in c[(c.condition == cond) & (c.cue_where == "first")].groupby("cue"):
            x = g.merge(ref[k2 + ["log_end"]], on=k2, suffixes=("", "_ref"))
            x["d"] = x.log_end - x.log_end_ref
            add(f"{cond}: first-turn cue '{cue}' vs none (plan end)", x, "d")
        # step cues: branched from the parent, compare steps >= cue_step
        for cue, g in c[(c.condition == cond) & (c.cue_where == "step")].groupby("cue"):
            x = g.merge(c[["conv_id", "log_late"]].rename(columns={"conv_id": "parent", "log_late": "log_late_ref"}), on="parent")
            x["d"] = x.log_late - x.log_late_ref
            add(f"{cond}: cue '{cue}' before step {cue_step} vs none (steps >= {cue_step})", x, "d")
    comp = pd.DataFrame(rows)
    # sanity: does the plan track the stated horizon?
    from scipy.stats import spearmanr
    b = base.dropna(subset=["log_end"])
    fit = np.polyfit(np.log10(b.h_years), b.log_end, 1) if len(b) > 2 else [math.nan, math.nan]
    adherence = pd.DataFrame([dict(
        conversations=len(c), step_turns=int((mt.kind == "step").sum()),
        steps_parsed=float(mt[mt.kind == "step"].h_step_years.notna().mean()),
        base_slope_end_vs_target=fit[0],
        base_spearman=spearmanr(np.log10(b.h_years), b.log_end).statistic if len(b) > 2 else math.nan,
        free_median_plan_end_years=float(c[(c.condition == "free") & c.cue.isna()].plan_end.median()))])
    return {"multiturn_comparisons": comp, "multiturn_adherence": adherence, "multiturn_conversations": c}
