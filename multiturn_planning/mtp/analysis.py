"""Analyses A-D on a capture run.

Rows are (conversation, turn). Turn t's pre-reply window (P0..P8) is read *before* reply t is written, so
for a step turn (t = 2..6) the reply's own horizon, step t-1, is not yet in the text there.

  A  persistence   y = log H_target. Probe fit on turn-1 rows, tested on turn-t rows (same position/layer).
  B  next step     y = log H_step of the reply about to be written, at that turn's pre-reply positions.
                   Compared with a text baseline that sees what's written: log H_target, previous step's
                   log H_step, step index. Reported as R2 of text, of activations, and of text + activations
                   (activations fit to the text baseline's cross-fitted residual).
  C  first turn    y_j = log H_step of step j (j = 1..5), from turn-1 pre-reply positions, beyond log H_target
                   (same residual scheme with a baseline on log H_target only).
  D  no horizon    B on the no-horizon conversations (the model chose every horizon), within-D and with the
                   probe trained on the horizon conversations.

All CV is grouped by scenario (GroupKFold), so a probe is always tested on scenarios it never saw.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from safetensors.numpy import load_file
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import GroupKFold

ALPHAS = np.logspace(-2, 6, 17)
LOG2 = np.log10(2)
N_FOLDS = 6
MIN_N = 5          # fewer valid samples than this in a cell: skipped


# ---- loading -----------------------------------------------------------------------------------------------

def load_run(run: Path):
    """index (one row per conversation-turn, with `row` = index into acts), acts [rows, L, P, d] float16,
    valid [rows, P], meta."""
    run = Path(run)
    df = pd.read_parquet(run / "index.parquet")
    meta = json.loads((run / "capture_meta.json").read_text())
    shards = sorted(run.glob("acts_*.safetensors"))
    parts = [load_file(str(s)) for s in shards]
    acts = np.concatenate([p["acts"] for p in parts])
    valid = np.concatenate([p["valid"] for p in parts]).astype(bool)
    offsets = np.cumsum([0] + [len(p["acts"]) for p in parts])
    df["row"] = [offsets[s] + r for s, r in zip(df["shard"], df["row_in_shard"])]
    df["log_h_step"] = np.log10(df["h_step_years"].astype(float))
    df["log_h_target"] = np.log10(df["h_target_years"].astype(float))
    return df, acts, valid, meta


def clean_conversations(df: pd.DataFrame) -> set[str]:
    """Conversations whose 5 step turns all parsed a horizon, carry the right step number, and weren't truncated."""
    st = df[df.kind == "step"]
    ok = st.groupby("conv_id").apply(
        lambda g: len(g) == 5 and g.h_step_years.notna().all() and (g.step_no == g.turn - 1).all()
        and not g.truncated.any())
    return set(ok[ok].index)


# ---- metrics and probes ---------------------------------------------------------------------------------------

def metrics(y, pred) -> dict:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    ss = ((y - y.mean()) ** 2).sum()
    err = np.abs(pred - y)
    return {"r2": 1 - ((y - pred) ** 2).sum() / ss if ss > 0 else np.nan,
            "rho": spearmanr(pred, y)[0] if np.std(pred) > 0 else np.nan,
            "mae_dec": err.mean(), "within2x": (err <= LOG2).mean(), "n": len(y)}


def _ridge(X, y):
    return RidgeCV(alphas=ALPHAS).fit(X, y)


def cv_predict(X, y, groups, folds=N_FOLDS) -> np.ndarray:
    """Out-of-fold ridge predictions, folds grouped by scenario."""
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=min(folds, len(set(groups)))).split(X, y, groups):
        pred[te] = _ridge(X[tr], y[tr]).predict(X[te])
    return pred


def cv_predict_linear(F, y, groups, folds=N_FOLDS) -> np.ndarray:
    """Same folds, plain least squares on a handful of text features."""
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=min(folds, len(set(groups)))).split(F, y, groups):
        pred[te] = LinearRegression().fit(F[tr], y[tr]).predict(F[te])
    return pred


def pc1_rho(X, y) -> float:
    Z = PCA(n_components=1).fit_transform(X - X.mean(0))[:, 0]
    return abs(spearmanr(Z, y)[0])


def feature(acts, rows, layer_i, pos_i) -> np.ndarray:
    return acts[rows, layer_i, pos_i].astype(np.float32)


# ---- A: persistence of H_target ----------------------------------------------------------------------------

def analysis_a(df, acts, valid, meta, positions, layers_i):
    hor = df[(df.condition == "horizon")]
    out = []
    first = hor[hor.turn == 1].set_index("conv_id")
    for li in layers_i:
        for pn in positions:
            pi = meta["positions"].index(pn)
            X1 = feature(acts, first.row.values, li, pi)
            y1, g1 = first.log_h_target.values, first.scenario.values
            folds = list(GroupKFold(n_splits=min(N_FOLDS, len(set(g1)))).split(X1, y1, g1))
            for t in sorted(hor.turn.unique()):
                cur = hor[hor.turn == t].set_index("conv_id").loc[first.index]
                ok = valid[cur.row.values, pi]
                if ok.sum() < MIN_N:                   # e.g. E on turns whose replies all hit max_new
                    continue
                Xt = feature(acts, cur.row.values, li, pi)
                pred_t = np.full(len(cur), np.nan)
                pred_in = np.full(len(cur), np.nan)
                for tr, te in folds:           # train on turn 1 of training scenarios, test on turn t of held-out ones
                    pred_t[te] = _ridge(X1[tr], y1[tr]).predict(Xt[te])
                    pred_in[te] = _ridge(Xt[tr][ok[tr]], y1[tr][ok[tr]]).predict(Xt[te]) if ok[tr].sum() > 5 else np.nan
                m = metrics(y1[ok], pred_t[ok])
                mi = metrics(y1[ok], pred_in[ok])
                out.append({"layer": meta["layers"][li], "pos": pn, "turn": t, **m,
                            "r2_same_turn": mi["r2"], "within2x_same_turn": mi["within2x"],
                            "pc1_rho": pc1_rho(Xt[ok], y1[ok])})
    return pd.DataFrame(out)


# ---- B / D: the step about to be written --------------------------------------------------------------------

def step_table(df, convs) -> pd.DataFrame:
    """One row per step turn with the text-baseline features."""
    st = df[(df.kind == "step") & df.conv_id.isin(convs)].sort_values(["conv_id", "turn"]).copy()
    st["k"] = st.turn - 1
    st["prev_log_h"] = st.groupby("conv_id").log_h_step.shift(1)
    return st


def text_features(st, with_target: bool) -> np.ndarray:
    k = st.k.values
    onehot = np.stack([(k == j).astype(float) for j in range(2, 6)], 1)           # k=1 is the reference
    prev = st.prev_log_h.values
    has_prev = ~np.isnan(prev)
    cols = [onehot, np.where(has_prev, prev, 0.0)[:, None], has_prev[:, None].astype(float)]
    if with_target:
        cols.append(st.log_h_target.values[:, None])
    return np.concatenate(cols, 1)


def analysis_b(df, acts, valid, meta, positions, layers_i, convs, condition, shuffle_seed=0):
    st = step_table(df, convs)
    st = st[st.condition == condition]
    if len(st) < MIN_N or st.scenario.nunique() < 2:
        return pd.DataFrame()
    y, g = st.log_h_step.values, st.scenario.values
    base = cv_predict_linear(text_features(st, condition == "horizon"), y, g)
    resid = y - base
    rng = np.random.default_rng(shuffle_seed)
    shuffled = resid.copy()
    for s in np.unique(g):                                  # shuffle within scenario: keeps group structure
        idx = np.where(g == s)[0]
        shuffled[idx] = rng.permutation(resid[idx])
    mt = metrics(y, base)
    out = []
    for li in layers_i:
        for pn in positions:
            pi = meta["positions"].index(pn)
            X = feature(acts, st.row.values, li, pi)
            pa = cv_predict(X, y, g)
            pr = cv_predict(X, resid, g)
            ps = cv_predict(X, shuffled, g)
            ma, mc = metrics(y, pa), metrics(y, base + pr)
            out.append({"condition": condition, "layer": meta["layers"][li], "pos": pn, "n": len(y),
                        "r2_text": mt["r2"], "r2_act": ma["r2"], "r2_text_plus_act": mc["r2"],
                        "delta_r2": mc["r2"] - mt["r2"],
                        "delta_r2_shuffled": metrics(y, base + ps)["r2"] - mt["r2"],
                        "within2x_text": mt["within2x"], "within2x_act": ma["within2x"],
                        "within2x_text_plus_act": mc["within2x"], "rho_act": ma["rho"],
                        "pc1_rho": pc1_rho(X, y)})
    return pd.DataFrame(out)


def analysis_d_transfer(df, acts, meta, positions, layers_i, convs):
    """Probe for the upcoming H_step trained on all horizon conversations, applied to the no-horizon ones."""
    st = step_table(df, convs)
    tr, te = st[st.condition == "horizon"], st[st.condition == "none"]
    if len(tr) < MIN_N or len(te) < MIN_N:
        return pd.DataFrame()
    out = []
    for li in layers_i:
        for pn in positions:
            pi = meta["positions"].index(pn)
            pred = _ridge(feature(acts, tr.row.values, li, pi), tr.log_h_step.values).predict(
                feature(acts, te.row.values, li, pi))
            out.append({"layer": meta["layers"][li], "pos": pn, **metrics(te.log_h_step.values, pred)})
    return pd.DataFrame(out)


# ---- C: does turn 1 hold the whole plan? --------------------------------------------------------------------------

def analysis_c(df, acts, meta, positions, layers_i, convs):
    hor = df[(df.condition == "horizon") & df.conv_id.isin(convs)]
    first = hor[hor.turn == 1].set_index("conv_id").sort_index()
    if len(first) < MIN_N or first.scenario.nunique() < 2:
        return pd.DataFrame()
    out = []
    for j in range(1, 6):
        yj = hor[hor.turn == j + 1].set_index("conv_id").loc[first.index].log_h_step.values
        g = first.scenario.values
        base = cv_predict_linear(first.log_h_target.values[:, None], yj, g)
        resid = yj - base
        mt = metrics(yj, base)
        for li in layers_i:
            for pn in positions:
                pi = meta["positions"].index(pn)
                X = feature(acts, first.row.values, li, pi)
                mc = metrics(yj, base + cv_predict(X, resid, g))
                out.append({"step": j, "layer": meta["layers"][li], "pos": pn, "n": len(yj),
                            "r2_target_only": mt["r2"], "r2_target_plus_act": mc["r2"],
                            "delta_r2": mc["r2"] - mt["r2"], "within2x_target_only": mt["within2x"],
                            "within2x_target_plus_act": mc["within2x"]})
    return pd.DataFrame(out)


# ---- behavior ---------------------------------------------------------------------------------------------------

def behavior(df, convs) -> dict:
    st = step_table(df, convs)
    hor = st[st.condition == "horizon"]
    last = hor[hor.k == 5]
    slope = np.polyfit(last.log_h_target, last.log_h_step, 1)[0] if len(last) > 2 else np.nan
    mono = st.groupby("conv_id").log_h_step.apply(lambda v: bool(np.all(np.diff(v.values) >= -1e-9)))
    within = (hor.log_h_step <= hor.log_h_target + np.log10(1.1)).mean()
    none_span = st[(st.condition == "none") & (st.k == 5)].log_h_step
    return {"n_clean_conversations": len(convs),
            "slope_last_step_vs_target": slope,
            "rho_last_step_vs_target": spearmanr(last.log_h_target, last.log_h_step)[0] if len(last) > 2 else np.nan,
            "frac_steps_within_target": within, "frac_monotone_plans": mono.mean(),
            "none_last_step_median_years": float(10 ** none_span.median()) if len(none_span) else np.nan,
            "none_last_step_iqr_decades": float(none_span.quantile(.75) - none_span.quantile(.25)) if len(none_span) else np.nan}
