# %% [markdown]
# # Horizon manifold explorer (notebook)
#
# Rotatable PC1–3 scatter of Qwen3-8B residual activations, one cell per
# (thinking mode, layer, token position). Uses the `pca3.json` exported by
# `analysis/export_pca3.py`. Change the widgets; the figure updates in place and
# keeps its camera. Drag to rotate, scroll to zoom.
#
# Data comes from a named **snapshot** under `snapshots/` (frozen derived data
# with provenance, made by `analysis/snapshot.py`). Set `SNAPSHOT` below; the
# comparison section at the end overlays two snapshots.

# %%
import json
from pathlib import Path

import ipywidgets as W
import numpy as np
import plotly.graph_objects as go

import sys
sys.path.insert(0, str(next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "analysis" / "snapshot.py").exists())))
from analysis.snapshot import list_snapshots, load_json

SNAPSHOT = "v1_noprefill"        # pick from the list printed below
for _name, _taken, _note, _runs in list_snapshots():
    print(f"{_name:20} {_taken}  {_note[:70]}")
DATA = load_json(SNAPSHOT, "pca3.json")
MODES = list(DATA["modes"])
try:
    DIM = load_json(SNAPSHOT, "feature_dimensionality.json")
except FileNotFoundError:
    DIM = {}

TICKS = [("30 s", 30 / 31_557_600), ("1 hr", 3600 / 31_557_600), ("1 day", 1 / 365), ("1 mo", 1 / 12),
         ("1 yr", 1), ("10 yr", 10), ("100 yr", 100), ("500 yr", 500)]
VIRIDIS = "Viridis"
DIVERGING = [[0, "#2c6fbb"], [0.5, "#b8bcc4"], [1, "#c0392b"]]

# %%
def traces(mode, layer, pos, color, show_controls):
    m = DATA["modes"][mode]
    cell = m["layers"][layer][pos]
    pts = m["points"]
    if cell is None:
        return [], None
    xyz = np.array(cell["xyz"])
    is_ctrl = np.array([p["lh"] is None for p in pts])
    hover = [f"{p['h']}<br>near: {p['near']}<br>far: {p['far']}" + (f"<br>answer: {p['ans']}" if p["ans"] else "")
             for p in pts]
    key = "lh" if color == "horizon" else "ratio"
    vals = [p[key] for p, c in zip(pts, is_ctrl) if not c]
    cbar = (dict(title="horizon", tickvals=[np.log10(v) for _, v in TICKS], ticktext=[t for t, _ in TICKS], thickness=12)
            if color == "horizon" else dict(title="log10(horizon / far delay)", thickness=12))
    out = [go.Scatter3d(
        x=xyz[~is_ctrl, 0], y=xyz[~is_ctrl, 1], z=xyz[~is_ctrl, 2], mode="markers", name="horizon prompts",
        text=[h for h, c in zip(hover, is_ctrl) if not c], hovertemplate="%{text}<extra></extra>",
        marker=dict(size=4.5, color=vals, colorscale=VIRIDIS if color == "horizon" else DIVERGING,
                    cmid=0 if color == "ratio" else None, colorbar=cbar, showscale=True))]
    if show_controls and is_ctrl.any():
        out.append(go.Scatter3d(
            x=xyz[is_ctrl, 0], y=xyz[is_ctrl, 1], z=xyz[is_ctrl, 2], mode="markers", name="no horizon",
            text=[h for h, c in zip(hover, is_ctrl) if c], hovertemplate="%{text}<extra></extra>",
            marker=dict(size=4.5, color="#8c8c8c")))
    return out, cell


def captured(mode, layer, pos, k=3):
    """Fraction of the linearly readable horizon signal inside the top-k PCs, or None."""
    c = DIM.get(mode, {}).get("cells", [[]])
    try:
        cell = c[layer][pos]
    except IndexError:
        return None
    if not cell or k not in cell["k"]:
        return None
    return cell["ratio"][cell["k"].index(k)], cell["full_r2"]


REGION_NAMES = {"suffix": "turn suffix", "think": "think", "pre": "pre", "prelabel": "pre-label",
                "answer": "answer"}


def region_of(m, pos):
    """Region label of a kept position; think positions carry their fraction,
    response tokens after the label read 'response'."""
    for name, (a, b) in m["regions"].items():
        if a <= pos < b:
            if name == "think":
                return "think " + m["tokens"][pos].split("@")[1].rstrip("*")
            if name == "answer" and pos > a:
                return "response"
            return REGION_NAMES[name]
    return "response"


def title(mode, layer, pos, cell):
    m = DATA["modes"][mode]
    tok = m["tokens"][pos]
    region = region_of(m, pos)
    if cell is None:
        return f"thinking {mode} · L{layer} · {tok!r} ({region}) · constant column"
    e, r = cell["evr"], cell["rho"]
    cap = captured(mode, layer, pos)
    tail = f"   top-3 PCs hold {cap[0]:.0%} of the horizon signal (full R² {cap[1]:.3f})" if cap else ""
    return (f"thinking {mode} · L{layer} · {tok!r} ({region})<br>"
            f"<sup>evr {e[0]:.2f}/{e[1]:.2f}/{e[2]:.2f}   |rho| with horizon {r[0]:.2f}/{r[1]:.2f}/{r[2]:.2f}{tail}</sup>")


# %%
fig = go.FigureWidget(layout=go.Layout(
    height=680, margin=dict(l=0, r=0, t=60, b=0), showlegend=False, uirevision="keep",
    scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3", aspectmode="cube",
               xaxis=dict(range=[-1.05, 1.05]), yaxis=dict(range=[-1.05, 1.05]), zaxis=dict(range=[-1.05, 1.05]))))

mode_w = W.ToggleButtons(options=MODES, description="thinking")
layer_w = W.IntSlider(value=DATA["modes"][MODES[0]]["meta"].get("probe_layer") or 11, min=0,
                      max=len(DATA["modes"][MODES[0]]["layers"]) - 1, description="layer", continuous_update=False)
pos_w = W.Dropdown(description="token")
color_w = W.ToggleButtons(options=[("horizon", "horizon"), ("horizon ÷ far delay", "ratio")], description="color")
ctrl_w = W.Checkbox(value=True, description="show no-horizon controls")


def fill_positions(*_):
    m = DATA["modes"][mode_w.value]
    opts = [(f"{i:>2}  {t!r}  ({region_of(m, i)})", i) for i, t in enumerate(m["tokens"])]
    keep = min(pos_w.value if pos_w.value is not None else (m["meta"].get("probe_pos") or 1), len(opts) - 1)
    pos_w.options = opts
    pos_w.value = keep
    layer_w.max = len(m["layers"]) - 1


dimfig = go.FigureWidget(layout=go.Layout(
    height=300, margin=dict(l=50, r=10, t=40, b=40), xaxis=dict(title="k = top principal components", type="log"),
    yaxis=dict(title="fraction", range=[0, 1.05]), legend=dict(orientation="h", y=-0.3)))


def update_dim():
    dimfig.data = ()
    c = DIM.get(mode_w.value, {}).get("cells")
    cell = c[layer_w.value][pos_w.value] if c and pos_w.value < len(c[layer_w.value]) else None
    if not cell:
        dimfig.layout.title = "no dimensionality results for this cell (run analysis/feature_dimensionality.py)"
        return
    k = cell["k"]
    with dimfig.batch_update():
        dimfig.add_trace(go.Scatter(x=k, y=cell["ratio"], mode="lines+markers", name="horizon R² in k dims ÷ full R²"))
        dimfig.add_trace(go.Scatter(x=k, y=cell["cos"], mode="lines+markers", name="|probe direction| inside k-dim subspace"))
        dimfig.add_trace(go.Scatter(x=k, y=cell["rho"], mode="lines+markers", name="Spearman |rho| of k-dim probe"))
        dimfig.add_trace(go.Scatter(x=k, y=cell["evr"], mode="lines", line=dict(dash="dash", color="gray"),
                                    name="cumulative explained variance (activations, not feature)"))
        dimfig.layout.title = dict(text=f"How many PCs hold the horizon?  full-space probe R² = {cell['full_r2']:.3f}",
                                   font=dict(size=13))


def update(*_):
    tr, cell = traces(mode_w.value, layer_w.value, pos_w.value, color_w.value, ctrl_w.value)
    with fig.batch_update():
        fig.data = ()
        for t in tr:
            fig.add_trace(t)
        fig.layout.title = dict(text=title(mode_w.value, layer_w.value, pos_w.value, cell), font=dict(size=13))
    update_dim()


mode_w.observe(lambda c: (fill_positions(), update()), names="value")
for w in (layer_w, pos_w, color_w, ctrl_w):
    w.observe(update, names="value")
fill_positions()
update()
W.VBox([W.HBox([mode_w, color_w]), W.HBox([layer_w, pos_w, ctrl_w]), fig, dimfig])

# %% [markdown]
# ## Quick comparisons
#
# Same token, every layer as a row of small multiples (static, for the write-up).

# %%
import plotly.subplots as sp

def layer_strip(mode, pos, layers=(1, 6, 11, 16, 21, 26, 31, 36)):
    m = DATA["modes"][mode]
    f = sp.make_subplots(rows=1, cols=len(layers), specs=[[{"type": "scene"}] * len(layers)],
                         subplot_titles=[f"L{l}" for l in layers], horizontal_spacing=0.01)
    for k, l in enumerate(layers, start=1):
        tr, _ = traces(mode, l, pos, "horizon", False)
        for t in tr:
            t.marker.showscale = False
            f.add_trace(t, row=1, col=k)
    f.update_layout(height=260, showlegend=False, margin=dict(l=0, r=0, t=30, b=0),
                    title=f"thinking {mode}, token {m['tokens'][pos]!r}")
    for k in range(1, len(layers) + 1):
        f.layout[f"scene{'' if k == 1 else k}"].update(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
    return f

layer_strip("off", 1)


# %% [markdown]
# ## Where the top-3 PCs hold the whole feature
#
# Fraction of the linearly readable horizon signal (full-space probe R²) that
# survives projection onto the top three principal components, per layer and
# token. Bright cells are where the 3D pictures above show everything; dark
# cells are where the horizon lives off the main axes of variation.

# %%
def captured_heatmap(mode, k=3):
    r = DIM[mode]
    M = [[(c["ratio"][c["k"].index(k)] if c and k in c["k"] else None) for c in row] for row in r["cells"]]
    f = go.Figure(go.Heatmap(z=M, x=[repr(t) for t in r["tokens"]], y=[f"L{l}" for l in range(len(M))],
                             zmin=0, zmax=1, colorscale="Viridis", colorbar=dict(title=f"fraction in top-{k}")))
    f.update_layout(height=620, title=f"thinking {mode}: fraction of horizon signal in the top-{k} PCs",
                    xaxis_title="token position", yaxis_title="layer", margin=dict(l=50, r=10, t=50, b=90))
    return f

for _m in DIM:
    captured_heatmap(_m).show()


# %% [markdown]
# ## Compare two snapshots
#
# Same cell (mode, layer, token) in two snapshots: the k-sweep curves overlaid,
# and the difference in the fraction captured by the top-3 PCs across every
# layer and token. Runs once a second snapshot exists (e.g. `v2_prefill` after
# the rerun). Token positions are matched by name, so layouts may differ.

# %%
A_NAME, B_NAME = SNAPSHOT, None          # set B_NAME, e.g. "v2_prefill"
_names = [n for n, *_ in list_snapshots()]
if B_NAME is None:
    B_NAME = next((n for n in _names if n != A_NAME), None)


def compare(a_name, b_name, mode="off", layer=11, token="\n"):
    A, B = load_json(a_name, "feature_dimensionality.json"), load_json(b_name, "feature_dimensionality.json")
    if mode not in A or mode not in B:
        print(f"mode {mode!r} missing in one snapshot"); return
    ta, tb = A[mode]["tokens"], B[mode]["tokens"]
    pa, pb = ta.index(token), tb.index(token)
    ca, cb = A[mode]["cells"][layer][pa], B[mode]["cells"][layer][pb]
    f = go.Figure()
    for nm, c in ((a_name, ca), (b_name, cb)):
        if c:
            f.add_trace(go.Scatter(x=c["k"], y=c["ratio"], mode="lines+markers", name=f"{nm}  (full R² {c['full_r2']:.3f})"))
    f.update_layout(height=320, title=f"thinking {mode} · L{layer} · {token!r}: horizon signal captured vs k",
                    xaxis=dict(title="k", type="log"), yaxis=dict(title="fraction", range=[0, 1.05]),
                    margin=dict(l=50, r=10, t=50, b=40))
    f.show()
    common = [t for t in ta if t in tb]
    D = []
    for l in range(min(len(A[mode]["cells"]), len(B[mode]["cells"]))):
        row = []
        for t in common:
            x, y = A[mode]["cells"][l][ta.index(t)], B[mode]["cells"][l][tb.index(t)]
            ok = x and y and 3 in x["k"] and 3 in y["k"]
            row.append((y["ratio"][y["k"].index(3)] - x["ratio"][x["k"].index(3)]) if ok else None)
        D.append(row)
    h = go.Figure(go.Heatmap(z=D, x=[repr(t) for t in common], y=[f"L{l}" for l in range(len(D))],
                             zmin=-0.5, zmax=0.5, colorscale="RdBu", zmid=0,
                             colorbar=dict(title=f"{b_name} − {a_name}")))
    h.update_layout(height=600, title=f"thinking {mode}: change in fraction of horizon signal in top-3 PCs",
                    xaxis_title="token (shared positions)", yaxis_title="layer", margin=dict(l=50, r=10, t=50, b=90))
    h.show()


if B_NAME:
    compare(A_NAME, B_NAME)
else:
    print("only one snapshot so far; take another with analysis/snapshot.py to compare")
