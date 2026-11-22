"""E21 condition set and edit specs, on cells fitted to synthetic activations (no model)."""

import numpy as np
import torch

from pipeline.s2_steering.experiments.ablate_steer import exp21_evaltypes_variants as E
from pipeline.s2_steering.experiments.ablate_steer.exp8_steering import delta_rows
from spar_horizon import steering as S

D = 24


def fake_cells():
    rng = np.random.default_rng(0)
    log_h = np.repeat(np.linspace(-2, 2, 9), 6)
    cells = {}
    for layer in (E.LAYER, *E.SWEEP):
        X = rng.normal(size=(len(log_h), D))
        X[:, layer % D] += 2 * log_h
        cells["off", layer] = S.fit_cell(X, log_h)
    return cells


def test_conditions_are_unique_and_cover_the_spec():
    names = [c[0] for c in E.steer_conditions()]
    assert len(names) == len(set(names))
    assert {"L24 curve -4", "L24 curve +1", "L24 random 0 +4", "L24 mean difference -2", "L24 line +2",
            "L16 curve +2", "L32 random 0 -2", "L22+24+26+28 curve +2"} <= set(names)
    ablate = [n for n, _ in E.ablate_conditions(D, np.eye(8, D))]
    assert len(ablate) == len(E.ABLATE_KS) * (1 + E.N_RANDOM_SUBSPACES)


def test_edit_specs_step_each_layer_with_its_own_cell():
    cells = fake_cells()
    log_h = np.array([-1.5, 0.3, 1.0])
    names, specs = E.edits_for(cells, "off", log_h, "cpu")
    assert names[0] == "base" and specs[0] is None and len(names) == len(specs) == len(set(names))
    by_name = dict(zip(names, specs))
    rows = [0, 2]
    # a single-layer step is E8's delta for that cell
    layers, make = by_name["L20 curve +2"]
    (edit,) = make(rows)
    assert layers == (20,)
    assert np.allclose(edit.delta.numpy(), delta_rows(cells["off", 20], "curve", 2, log_h)[rows], atol=1e-6)
    # the stacked step adds each layer's own delta
    layers, make = by_name["L22+24+26+28 curve -2"]
    assert layers == E.STACK
    for layer, edit in zip(layers, make(rows)):
        assert np.allclose(edit.delta.numpy(), delta_rows(cells["off", layer], "curve", -2, log_h)[rows], atol=1e-6)
    # removal at L24 sets the horizon subspace to the bank mean
    layer, make = by_name["L24 ablate horizon k2"]
    cell = cells["off", E.LAYER]
    x = torch.randn(3, D)
    out = make(rows)(x, None, None).numpy()
    assert layer == E.LAYER
    assert np.allclose((out - cell["mean"]) @ cell["basis"][:2].T, 0, atol=1e-5)


def planted_results():
    """Two types; the L24 curve moves near log-odds by -alpha (one logit per decade), the stack by
    -2 alpha, everything else does nothing; the horizon subspace removes half the contrast."""
    from pipeline.s3_analysis.eval_types_variants import SWEEP
    names = [c[0] for c in E.steer_conditions()] + [n for n, _ in E.ablate_conditions(D, np.eye(8, D))]
    meta_rows, rows = [], []
    for t in ("ta", "tb"):
        for i in range(6):
            regime = "near" if i % 2 == 0 else "both"
            for variant in ("base", "swap"):
                uid = f"{t}_{i:03d}_{variant}"
                base = 5.0 if regime == "near" else -5.0
                meta_rows.append({"sample_uid": uid, "type_id": t, "variant": variant, "twin_id": f"{t}_{i:03d}",
                                  "regime": regime, "near_label": "a", "horizon": 0.5 if regime == "near" else 50.0})
                row = {"sample_uid": uid, "base": base}
                for n in names:
                    alpha = int(n.rsplit(" ", 1)[1]) if n.split()[-1][0] in "+-" else 0
                    if n.startswith("L24 curve"):
                        row[n] = base - alpha
                    elif n.startswith("L22+24+26+28 curve"):
                        row[n] = base - 2 * alpha
                    elif "ablate horizon" in n:
                        row[n] = base / 2
                    else:
                        row[n] = base
                rows.append(row)
    curve_range = {str(layer): [-6.0, 2.7] for layer in SWEEP}
    results = {"modes": {"off": {"rows": rows, "curve_range": curve_range}}}
    return results, {"rows": meta_rows}


def test_variants_analysis_recovers_planted_effects():
    from pipeline.s3_analysis.eval_types_variants import analyse, markdown
    out = analyse(*planted_results())
    m = out["modes"]["off"]
    a = m["types"]["ta"]
    assert a["baseline"]["gate"] and a["baseline"]["contrast"] == 10.0
    assert [a["dose"][f"curve {s}"]["toward"]["mean"] for s in (1, 2, 4)] == [0.1, 0.2, 0.4]
    assert a["dose"]["random 0 4"]["toward"]["mean"] == 0.0 and a["stacked"]["toward"]["mean"] == 0.4
    assert a["direction"]["mean difference"]["toward"]["mean"] == 0.0
    assert a["layer"][24]["curve"]["toward"]["mean"] == 0.2 and a["layer"][16]["curve"]["toward"]["mean"] == 0.0
    # near rows at log10(0.5) = -0.3 stop at the curve's end, 2.7: they move 3.0 of the 4 decades
    assert a["effective_step"][4] == round((3.0 + 4.0) / 2, 2)
    assert all(np.isclose(a["removal"][k]["horizon"][0], 0.5) and a["removal"][k]["beats_random"] for k in E.ABLATE_KS)
    R = m["readouts"]
    assert R["dose"]["ratio 4/2"] == 2.0 and R["dose"]["clean"]
    assert R["direction"]["curve minus mean difference"]["mean"] == 0.2
    assert R["layer"]["best layer per type"] == {"ta": 24, "tb": 24}
    assert R["removal"][4] == {"types beating random": 2, "of": 2}
    assert "ta" in markdown(out)
