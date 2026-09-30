"""Plant a hidden per-conversation plan offset z in fake activations and check A-D find what was planted.

log H_step(k) = log H_target + log10(k/5) + z. The text baseline sees H_target, k and the previous step (which
reveals z from step 2 on), so only the pre-reply position that carries z (P8) should add R2 over text,
mostly at step 1; P0 carries only noise.
"""

import numpy as np
import pandas as pd

from mtp import analysis as an

POS = [f"P{i}" for i in range(9)] + ["U", "H", "V", "E"]


def fake_run(seed=0, d=64):
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(3, d))
    rows, acts = [], []
    for sc in range(12):
        for cond, targets in (("horizon", np.log10([1 / 52, 1 / 12, .25, 1, 3, 10, 25, 50])), ("none", [np.nan] * 6)):
            for i, lt in enumerate(targets):
                cid = f"s{sc}_{cond}_{i}"
                z = rng.normal(0, 0.3)
                base_t = lt if cond == "horizon" else rng.normal(0.5, 0.5)
                for t in range(1, 8):
                    k = t - 1
                    lh = base_t + np.log10(k / 5) + z if 1 <= k <= 5 else np.nan
                    a = rng.normal(0, 1, size=(1, 1, len(POS), d))   # [row, layer, pos, d]
                    if cond == "horizon":
                        a[0, 0, :9] += 2 * lt * dirs[0]                       # H_target at every pre position
                    a[0, 0, 8] += 3 * z * dirs[1]                             # the hidden plan only at P8
                    if 1 <= k <= 5:
                        a[0, 0, 12] += 3 * lh * dirs[2]
                    acts.append(a.astype(np.float16))
                    rows.append({"conv_id": cid, "scenario": f"s{sc}", "condition": cond, "turn": t,
                                 "kind": "outline" if t == 1 else ("done" if t == 7 else "step"),
                                 "step_no": k if 1 <= k <= 5 else None, "truncated": False,
                                 "h_step_years": 10 ** lh if 1 <= k <= 5 else np.nan,
                                 "h_target_years": 10 ** lt if cond == "horizon" else np.nan,
                                 "row": len(acts) - 1})
    df = pd.DataFrame(rows)
    df["log_h_step"] = np.log10(df.h_step_years)
    df["log_h_target"] = np.log10(df.h_target_years)
    return df, np.concatenate(acts), np.ones((len(acts), len(POS)), bool), {"positions": POS, "layers": [0]}


def test_b_finds_the_planted_plan_only_where_planted():
    df, acts, valid, meta = fake_run()
    convs = an.clean_conversations(df)
    assert len(convs) == df.conv_id.nunique()
    B = an.analysis_b(df, acts, valid, meta, ["P0", "P8"], [0], convs, "horizon").set_index("pos")
    assert B.loc["P8", "delta_r2"] > 0.02 > B.loc["P0", "delta_r2"]
    assert abs(B.loc["P8", "delta_r2_shuffled"]) < B.loc["P8", "delta_r2"]


def test_a_reads_target_at_every_turn_and_c_sees_the_plan():
    df, acts, valid, meta = fake_run(1)
    A = an.analysis_a(df, acts, valid, meta, ["P0"], [0])
    assert (A.r2 > 0.8).all()
    C = an.analysis_c(df, acts, meta, ["P0", "P8"], [0], an.clean_conversations(df)).groupby("pos").delta_r2.mean()
    assert C["P8"] > 0.03 > C["P0"]          # z is ~6% of the variance of log H_step here (target spans 3.4 decades)


def test_unparsed_step_drops_only_that_step():
    df, acts, valid, meta = fake_run(2)
    hit = df[(df.kind == "step") & (df.turn == 6) & (df.condition == "horizon")].index[:10]   # 10 step-5 "Ongoing"
    df.loc[hit, ["h_step_years", "log_h_step"]] = np.nan
    convs = an.clean_conversations(df)
    assert len(convs) == df.conv_id.nunique()                          # conversations kept
    st = an.step_table(df, convs)
    assert len(st) == (df.kind == "step").sum() - 10 and st.log_h_step.notna().all()
    assert not an.analysis_b(df, acts, valid, meta, ["P8"], [0], convs, "horizon").empty
    C = an.analysis_c(df, acts, meta, ["P8"], [0], convs)
    assert set(C.step) == {1, 2, 3, 4, 5} and C[C.step == 5].n.iloc[0] == (C[C.step == 1].n.iloc[0] - 10)
