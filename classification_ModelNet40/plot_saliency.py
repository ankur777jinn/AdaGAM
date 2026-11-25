"""
3D Plotly Visualization of Per-Point Adaptation Saliency Maps.

Loads .npz files produced by saliency_map.py and generates side-by-side
interactive 3D HTML plots (clean vs corrupted) colored by |Delta alpha|,
plus static PNG exports if kaleido is installed.

Usage:
    python plot_saliency.py --input saliency_output/ --out saliency_figures/
"""

import argparse
import glob
import os

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def make_side_by_side_figure(pts_clean, sal_clean, pts_corr, sal_corr, corr_name, sev, obj_idx):
    cmax = float(max(sal_clean.max(), sal_corr.max(), 1e-6))

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=(
            f"Clean (mean |Δα|={sal_clean.mean():.4f})",
            f"{corr_name} sev={sev} (mean |Δα|={sal_corr.mean():.4f})",
        ),
    )

    fig.add_trace(
        go.Scatter3d(
            x=pts_clean[:, 0], y=pts_clean[:, 1], z=pts_clean[:, 2],
            mode="markers",
            marker=dict(size=3, color=sal_clean, colorscale="Viridis", cmin=0, cmax=cmax),
            name="Clean",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=pts_corr[:, 0], y=pts_corr[:, 1], z=pts_corr[:, 2],
            mode="markers",
            marker=dict(
                size=3, color=sal_corr, colorscale="Viridis", cmin=0, cmax=cmax,
                colorbar=dict(title="|Δα|", len=0.75),
            ),
            name=corr_name,
        ),
        row=1, col=2,
    )

    cam = dict(eye=dict(x=1.5, y=1.5, z=1.0))
    axis_off = dict(visible=False)
    scene_cfg = dict(
        xaxis=axis_off, yaxis=axis_off, zaxis=axis_off,
        aspectmode="data", camera=cam,
    )
    fig.update_layout(
        title=f"AdaGAM Adaptation Saliency — Object #{obj_idx} ({corr_name})",
        scene=scene_cfg,
        scene2=scene_cfg,
        width=1100, height=550,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="saliency_output", help="directory with .npz files")
    p.add_argument("--out", default="saliency_figures", help="output directory for HTML/PNG")
    p.add_argument("--max_objects", default=4, type=int, help="max objects per corruption to plot")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    clean_path = os.path.join(args.input, "clean.npz")
    if not os.path.isfile(clean_path):
        raise FileNotFoundError(f"Missing {clean_path}; include 'clean' in saliency_map.py --corruptions")

    clean_data = np.load(clean_path)
    corr_files = sorted(f for f in glob.glob(os.path.join(args.input, "*.npz"))
                        if os.path.basename(f) != "clean.npz")

    has_kaleido = False
    try:
        import kaleido  # noqa: F401
        has_kaleido = True
    except ImportError:
        pass

    for fpath in corr_files:
        cdata = np.load(fpath)
        cname = str(cdata["corruption"])
        sev = int(cdata["severity"])
        n_obj = min(args.max_objects, len(cdata["points"]))
        for i in range(n_obj):
            fig = make_side_by_side_figure(
                clean_data["points"][i], clean_data["saliency"][i],
                cdata["points"][i], cdata["saliency"][i],
                cname, sev, int(cdata["sample_indices"][i]),
            )
            stem = os.path.join(args.out, f"saliency_{cname}_sev{sev}_obj{i}")
            fig.write_html(stem + ".html")
            if has_kaleido:
                fig.write_image(stem + ".png", scale=2)
        print(f"Plotted {n_obj} objects for {cname} (sev={sev}) -> {args.out}")


if __name__ == "__main__":
    main()
