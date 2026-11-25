"""Side-by-side 3D views (Plotly HTML, optional PNG via kaleido) of the adaptation saliency:
left = clean cloud, right = corrupted cloud, both coloured by per-point stage-1 |d_alpha| on a shared
colour scale. Titles show the true label, the prediction, and the prediction with adaptation off.

    python plot_saliency.py --input saliency_output/ --out saliency_figures/ [--png]
"""
import argparse
import glob
import os

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data import SHAPE_NAMES


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def name(i):
    return SHAPE_NAMES[int(i)] if 0 <= int(i) < len(SHAPE_NAMES) else str(int(i))


def scene(fig, d, j, col, cmax):
    pts, sal = d["points"][j], d["saliency"][j]
    fig.add_trace(go.Scatter3d(
        x=pts[:, 0], y=pts[:, 2], z=pts[:, 1], mode="markers",
        marker=dict(size=2.5, color=sal, colorscale="Viridis", cmin=0, cmax=cmax,
                    colorbar=dict(title="|Δα|", x=1.02) if col == 2 else None, showscale=col == 2),
        hovertemplate="|Δα|=%{marker.color:.4f}<extra></extra>"), row=1, col=col)


def figure(clean, corr, j, cmax):
    lab = name(corr["labels"][j])
    t1 = f"clean: pred {name(clean['preds'][j])} (off: {name(clean['preds_counterfactual'][j])})"
    t2 = f"{corr['split']}: pred {name(corr['preds'][j])} (off: {name(corr['preds_counterfactual'][j])})"
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]], subplot_titles=(t1, t2))
    scene(fig, clean, j, 1, cmax)
    scene(fig, corr, j, 2, cmax)
    axis = dict(showticklabels=False, title="", showgrid=False, zeroline=False, showbackground=False)
    for s in ("scene", "scene2"):
        fig.layout[s].update(xaxis=axis, yaxis=axis, zaxis=axis, aspectmode="data")
    fig.update_layout(title=f"Sample {int(corr['sample_indices'][j])} · true class: {lab} · colour = stage-1 |Δα| (AdaGAM adaptation)",
                      height=520, width=1100, margin=dict(l=0, r=0, t=70, b=0), showlegend=False)
    return fig


def main(argv=None):
    p = argparse.ArgumentParser("Plot adaptation saliency")
    p.add_argument("--input", default="saliency_output")
    p.add_argument("--out", default="saliency_figures")
    p.add_argument("--png", action="store_true", help="also export PNG (needs `pip install kaleido`)")
    p.add_argument("--max_samples", type=int, default=0)
    a = p.parse_args(argv)
    clean_path = os.path.join(a.input, "saliency_clean.npz")
    if not os.path.exists(clean_path):
        raise SystemExit(f"{clean_path} missing - run saliency_map.py first")
    clean = load(clean_path)
    files = sorted(f for f in glob.glob(os.path.join(a.input, "saliency_*.npz")) if not f.endswith("saliency_clean.npz"))
    os.makedirs(a.out, exist_ok=True)
    links = []
    for f in files:
        corr = load(f)
        corr["split"] = str(corr["split"])
        if not np.array_equal(clean["sample_indices"], corr["sample_indices"]):
            raise SystemExit(f"{f}: sample_indices differ from saliency_clean.npz - re-run saliency_map.py for both")
        if not np.array_equal(clean["labels"], corr["labels"]):
            raise SystemExit(f"{f}: labels differ from the clean file for the same indices - files are misaligned")
        cmax = float(np.percentile(np.concatenate([clean["saliency"].ravel(), corr["saliency"].ravel()]), 99)) or 1e-6
        n = len(corr["sample_indices"]) if not a.max_samples else min(a.max_samples, len(corr["sample_indices"]))
        for j in range(n):
            fig = figure(clean, corr, j, cmax)
            stem = f"{corr['split']}_sample{int(corr['sample_indices'][j])}"
            fig.write_html(os.path.join(a.out, stem + ".html"), include_plotlyjs="cdn")
            links.append(stem)
            if a.png:
                try:
                    fig.write_image(os.path.join(a.out, stem + ".png"), scale=2)
                except Exception as e:  # kaleido missing or broken
                    print(f"  PNG export skipped ({e}); `pip install kaleido` to enable")
                    a.png = False
    with open(os.path.join(a.out, "index.html"), "w", encoding="utf-8") as fh:
        fh.write("<html><body><h2>AdaGAM adaptation saliency</h2><ul>")
        fh.write("".join(f'<li><a href="{s}.html">{s}</a></li>' for s in links))
        fh.write("</ul></body></html>")
    print(f"==> wrote {len(links)} figures + index.html to {a.out}/")


if __name__ == "__main__":
    main()
