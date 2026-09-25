#!/usr/bin/env python
"""Interactive 3-D view of a cell: samples and the bin-centroid path in (PC1, PC2, PC3).

Writes one standalone HTML per cell (open in a browser; drag to rotate, scroll to zoom).
  --basis centroid  (default) PCA fit on the horizon-bin centroids, all samples projected — the
                    same basis as the 2-D path figures from geometry_path.py
  --basis sample    PCA fit on all horizon samples — the basis of analyze.py's cell scatters
Buttons switch the point coloring between log horizon and the model's choice.

Usage: plot3d.py RUN_DIR OUT_DIR [--cells 22:R0,37:T3] [--basis centroid|sample] [--max-points 4000]
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from ptm.analysis import cell_name, COLOR_SHORT, COLOR_LONG, COLOR_NULL
from ptm.store import RunData

ap = argparse.ArgumentParser()
ap.add_argument("run_dir"); ap.add_argument("out_dir")
ap.add_argument("--cells", default="22:R0,37:T3")
ap.add_argument("--basis", choices=["centroid", "sample"], default="centroid")
ap.add_argument("--max-points", type=int, default=4000)
a = ap.parse_args()
run = RunData(a.run_dir); df = run.index; out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
has_h = df["horizon_years"].notna().to_numpy()
bins = df.loc[has_h].groupby("horizon_text")["horizon_years"].first().sort_values()
labels, years = bins.index.tolist(), bins.to_numpy()
ht = df["horizon_text"].to_numpy()

for cell in a.cells.split(","):
    layer, pos = cell.split(":"); layer = int(layer)
    X = run.get(layer, pos); valid = run.valid_mask(pos)
    C = np.stack([X[valid & has_h & (ht == h)].mean(0) for h in labels])
    pca = PCA(3, random_state=0).fit(C if a.basis == "centroid" else X[valid & has_h])
    ev = pca.explained_variance_ratio_
    P = pca.transform(X[valid]); Cp = pca.transform(C)
    d = df[valid].reset_index(drop=True)
    if len(d) > a.max_points:
        keep = np.random.default_rng(0).choice(len(d), a.max_points, replace=False); P, d = P[keep], d.iloc[keep].reset_index(drop=True)
    hy = d["horizon_years"].to_numpy(dtype=float); null = np.isnan(hy)
    hover = [f"{r.horizon_text or 'no horizon'}<br>choice: {'short' if r.chose_short == True else 'long' if r.chose_short == False else 'n/a'}"
             f"<br>p_short: {r.p_short:.3f}<br>options: {r.short_delay_text} / {r.long_delay_text}<br>{r.sample_uid}" for r in d.itertuples()]
    choice_color = np.where(d["chose_short"] == True, COLOR_SHORT, np.where(d["chose_short"] == False, COLOR_LONG, COLOR_NULL))

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=P[~null, 0], y=P[~null, 1], z=P[~null, 2], mode="markers", name="samples",
                               marker=dict(size=2.5, color=np.log10(hy[~null]), colorscale="Viridis", opacity=0.75,
                                           colorbar=dict(title="log10 horizon (y)", thickness=12)),
                               text=[t for t, n in zip(hover, null) if not n], hoverinfo="text"))
    fig.add_trace(go.Scatter3d(x=P[null, 0], y=P[null, 1], z=P[null, 2], mode="markers", name="no horizon",
                               marker=dict(size=2.5, color=COLOR_NULL, opacity=0.75), text=[t for t, n in zip(hover, null) if n], hoverinfo="text"))
    fig.add_trace(go.Scatter3d(x=Cp[:, 0], y=Cp[:, 1], z=Cp[:, 2], mode="lines+markers+text", name="bin-centroid path",
                               line=dict(color="black", width=4), marker=dict(size=5, color=np.log10(years), colorscale="Viridis", line=dict(color="black", width=1)),
                               text=[l if i in (0, len(labels) // 2, len(labels) - 1) else "" for i, l in enumerate(labels)], textposition="top center",
                               hovertext=[f"{l} centroid" for l in labels], hoverinfo="text"))
    # coloring toggle: horizon vs choice (applies to the sample trace)
    fig.update_layout(
        updatemenus=[dict(type="buttons", direction="left", x=0.0, y=1.08, showactive=True, buttons=[
            dict(label="color: horizon", method="restyle",
                 args=[{"marker.color": [np.log10(hy[~null])], "marker.colorscale": ["Viridis"], "marker.showscale": [True]}, [0]]),
            dict(label="color: choice (teal short / orange long)", method="restyle",
                 args=[{"marker.color": [choice_color[~null]], "marker.showscale": [False]}, [0]]),
        ])],
        scene=dict(xaxis_title=f"{a.basis} PC1 ({ev[0]:.2f})", yaxis_title=f"{a.basis} PC2 ({ev[1]:.2f})", zaxis_title=f"{a.basis} PC3 ({ev[2]:.2f})",
                   aspectmode="data"),
        title=f"{run.meta['model_name']}  {cell_name(run, layer, pos)}   basis: {a.basis} PCA   n={len(d)}",
        legend=dict(x=0.8, y=0.95), margin=dict(l=0, r=0, t=60, b=0), height=800,
    )
    path = out / f"view3d_{a.basis}_L{layer:02d}_{pos}.html"
    fig.write_html(str(path), include_plotlyjs="directory", full_html=True)
    print(f"wrote {path}  ({path.stat().st_size // 1024} KB; plotly.min.js shared in {out})")
