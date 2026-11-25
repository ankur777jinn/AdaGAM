"""
AdaGAM Live Inference Screening & Saliency Explorer (Gradio + Plotly 3D).

Features:
  1. Interactive 3D Plotly side-by-side viewer: Clean vs. Corrupted point cloud
     colored by Stage-1 per-point adaptation saliency |Delta alpha|.
  2. Live Counterfactual Ablation Switch: toggle AdaGAM (Delta alpha, Delta beta)
     ON vs. OFF (zeroed) on the exact same corrupted cloud and trained weights.
  3. Top-5 class confidence comparison & Stage 1-4 |Delta alpha| telemetry bars.
  4. Works with ModelNet-C clean.h5, custom .npy uploads, or a built-in fallback
     shape if no dataset is present.

Usage:
    python demo_app.py --adagam_ckpt checkpoints/pointMLP-adagam-42 \
        --base_ckpt checkpoints/pointMLP-base-42 --data data/modelnet_c
"""

import argparse
import os

import gradio as gr
import h5py
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import torch.nn.functional as F

from corruptions import ALL_CORRUPTIONS, _FUNCS as CORRUPTION_FUNCS
from eval_modelnet_c import load_model
from saliency_map import compute_point_saliency

MODELNET40_CLASSES = [
    "airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl", "car",
    "chair", "cone", "cup", "curtain", "desk", "door", "dresser", "flower_pot",
    "glass_box", "guitar", "keyboard", "lamp", "laptop", "mantel", "monitor",
    "night_stand", "person", "piano", "plant", "radio", "range_hood", "sink",
    "sofa", "stairs", "stool", "table", "tent", "toilet", "tv_stand", "vase",
    "wardrobe", "xbox",
]


def _make_fallback_cloud(n_pts=1024, seed=42):
    rng = np.random.RandomState(seed)
    t = rng.uniform(0, 2 * np.pi, n_pts)
    p = rng.uniform(0, 2 * np.pi, n_pts)
    R, r = 0.65, 0.22
    x = (R + r * np.cos(p)) * np.cos(t)
    y = (R + r * np.cos(p)) * np.sin(t)
    z = r * np.sin(p)
    return np.stack([x, y, z], axis=-1).astype("float32")


def _load_sample_bank(data_dir, max_samples=50):
    clean_h5 = os.path.join(data_dir, "clean.h5") if data_dir else ""
    if clean_h5 and os.path.isfile(clean_h5):
        with h5py.File(clean_h5, "r") as f:
            pts = f["data"][:max_samples].astype("float32")
            lbl = f["label"][:max_samples].astype("int64").reshape(-1)
        return pts, lbl
    return np.stack([_make_fallback_cloud(1024, seed=i) for i in range(5)], axis=0), np.zeros(5, dtype=np.int64)


@torch.no_grad()
def _infer_probs(net, pts_np, device, disable_adapt=False):
    adagam_mods = [m for m in net.modules() if m.__class__.__name__ == "AdaGAM"]
    for m in adagam_mods:
        m.disable_adaptation = disable_adapt
        m.record_stats = not disable_adapt
    x = torch.from_numpy(pts_np[None, ...]).to(device).permute(0, 2, 1)
    logits = net(x)
    probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
    stage_dalpha = [m.last_stats["d_alpha_abs"] if m.last_stats else 0.0 for m in adagam_mods]
    for m in adagam_mods:
        m.disable_adaptation = False
        m.record_stats = False
    return probs, stage_dalpha


def _build_3d_fig(pts_clean, pts_corr, saliency, corr_name, severity):
    cmax = float(max(saliency.max(), 1e-6))
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=(
            "Clean Input (1024 pts)",
            f"Screened Input ({corr_name}, sev={severity}) — |Δα| Saliency",
        ),
    )
    fig.add_trace(
        go.Scatter3d(
            x=pts_clean[:, 0], y=pts_clean[:, 1], z=pts_clean[:, 2],
            mode="markers",
            marker=dict(size=3, color="#60a5fa", opacity=0.85),
            name="Clean",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=pts_corr[:, 0], y=pts_corr[:, 1], z=pts_corr[:, 2],
            mode="markers",
            marker=dict(
                size=3.2, color=saliency, colorscale="Turbo", cmin=0, cmax=cmax,
                colorbar=dict(title="|Δα|", len=0.75), opacity=0.9,
            ),
            name="Screened",
        ),
        row=1, col=2,
    )
    cam = dict(eye=dict(x=1.45, y=1.45, z=0.95))
    axis_off = dict(visible=False)
    scene_cfg = dict(xaxis=axis_off, yaxis=axis_off, zaxis=axis_off, aspectmode="data", camera=cam)
    fig.update_layout(
        scene=scene_cfg, scene2=scene_cfg,
        height=470, margin=dict(l=10, r=10, t=40, b=10),
        template="plotly_dark",
    )
    return fig


def _build_prob_fig(probs_on, probs_off, probs_base=None):
    top_idx = np.argsort(probs_on)[-5:][::-1]
    labels = [MODELNET40_CLASSES[i] for i in top_idx][::-1]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=[probs_on[i] * 100 for i in top_idx][::-1],
        name="AdaGAM ON", orientation="h", marker_color="#22c55e",
    ))
    fig.add_trace(go.Bar(
        y=labels, x=[probs_off[i] * 100 for i in top_idx][::-1],
        name="AdaGAM OFF (Δα=0, Δβ=0)", orientation="h", marker_color="#f59e0b",
    ))
    if probs_base is not None:
        fig.add_trace(go.Bar(
            y=labels, x=[probs_base[i] * 100 for i in top_idx][::-1],
            name="Static PointMLP", orientation="h", marker_color="#64748b",
        ))
    fig.update_layout(
        barmode="group", title="Top-5 Class Confidence (%)",
        xaxis_title="Confidence (%)", xaxis_range=[0, 100],
        height=300, margin=dict(l=10, r=10, t=40, b=30),
        template="plotly_dark", legend=dict(orientation="h", y=-0.22),
    )
    return fig


def _build_stage_fig(stage_dalpha):
    stages = [f"Stage {i + 1}" for i in range(len(stage_dalpha))]
    fig = go.Figure(go.Bar(
        x=stages, y=stage_dalpha,
        marker_color=["#38bdf8", "#818cf8", "#c084fc", "#f472b6"][:len(stages)],
        text=[f"{v:.4f}" for v in stage_dalpha], textposition="auto",
    ))
    fig.update_layout(
        title="Stage-wise Adaptation Magnitude (mean |Δα|)",
        yaxis_title="|Δα|", height=300,
        margin=dict(l=10, r=10, t=40, b=30), template="plotly_dark",
    )
    return fig


def create_app(adagam_net, base_net, sample_pts, sample_lbl, device):
    def run_screening(sample_idx, npy_file, corr_name, severity, adapt_on):
        if npy_file is not None:
            pts_clean = np.load(npy_file.name).astype("float32")[:, :3]
            gt_name = "custom .npy"
        else:
            idx = int(sample_idx) % len(sample_pts)
            pts_clean = sample_pts[idx].copy()
            gt_name = MODELNET40_CLASSES[int(sample_lbl[idx])]

        if corr_name == "clean":
            pts_corr = pts_clean.copy()
        else:
            pts_corr = CORRUPTION_FUNCS[corr_name](pts_clean.copy(), int(severity)).astype("float32")

        probs_on, stage_dalpha = _infer_probs(adagam_net, pts_corr, device, disable_adapt=False)
        probs_off, _ = _infer_probs(adagam_net, pts_corr, device, disable_adapt=True)
        probs_base = _infer_probs(base_net, pts_corr, device)[0] if base_net is not None else None

        if adapt_on:
            saliency, _ = compute_point_saliency(adagam_net, pts_corr[None, ...], device)
            saliency = saliency[0]
        else:
            saliency = np.zeros(len(pts_corr), dtype=np.float32)

        pred_on = f"{MODELNET40_CLASSES[int(probs_on.argmax())]} ({probs_on.max() * 100:.1f}%)"
        pred_off = f"{MODELNET40_CLASSES[int(probs_off.argmax())]} ({probs_off.max() * 100:.1f}%)"
        summary = (
            f"### Live Screening Telemetry\n"
            f"- **Ground Truth:** `{gt_name}` | **Corruption:** `{corr_name}` (sev `{severity}`)\n"
            f"- **AdaGAM ON:** `{pred_on}`\n"
            f"- **Counterfactual OFF (Δα=0, Δβ=0):** `{pred_off}`\n"
        )
        if probs_base is not None:
            pred_base = f"{MODELNET40_CLASSES[int(probs_base.argmax())]} ({probs_base.max() * 100:.1f}%)"
            summary += f"- **Static PointMLP Baseline:** `{pred_base}`\n"

        return (
            _build_3d_fig(pts_clean, pts_corr, saliency, corr_name, severity),
            summary,
            _build_prob_fig(probs_on, probs_off, probs_base),
            _build_stage_fig(stage_dalpha if adapt_on else [0.0] * len(stage_dalpha)),
        )

    with gr.Blocks(title="AdaGAM Live Inference Screening") as demo:
        gr.Markdown("# AdaGAM — Live 3D Robustness Screening & Adaptation Saliency")
        gr.Markdown(
            "Inspect per-point `|Δα|` adaptation saliency under ModelNet-C corruptions and "
            "toggle `(Δα, Δβ)` ON/OFF live on the same trained weights."
        )
        with gr.Row():
            with gr.Column(scale=1):
                sample_idx = gr.Slider(0, max(0, len(sample_pts) - 1), value=0, step=1, label="Test Sample Index")
                npy_file = gr.File(label="Or Upload Custom .npy [N, 3]", file_types=[".npy"])
                corr_name = gr.Dropdown(["clean"] + ALL_CORRUPTIONS, value="jitter", label="Corruption Type")
                severity = gr.Slider(0, 4, value=3, step=1, label="Corruption Severity (0–4)")
                adapt_on = gr.Checkbox(value=True, label="Enable AdaGAM Modulation (Δα, Δβ)")
                run_btn = gr.Button("Run Live Screening", variant="primary")
                summary_md = gr.Markdown()
            with gr.Column(scale=3):
                fig_3d = gr.Plot(label="3D Point Cloud & Adaptation Saliency")
                with gr.Row():
                    fig_prob = gr.Plot(label="Class Confidence Comparison")
                    fig_stage = gr.Plot(label="Stage-wise |Δα| Telemetry")

        inputs = [sample_idx, npy_file, corr_name, severity, adapt_on]
        outputs = [fig_3d, summary_md, fig_prob, fig_stage]
        run_btn.click(fn=run_screening, inputs=inputs, outputs=outputs)
        demo.load(fn=run_screening, inputs=inputs, outputs=outputs)

    return demo


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adagam_ckpt", required=True, help="trained AdaGAM checkpoint directory")
    p.add_argument("--base_ckpt", default=None, help="optional static PointMLP checkpoint directory")
    p.add_argument("--data", default="data/modelnet_c", help="ModelNet-C directory")
    p.add_argument("--which", default="last", choices=["last", "best"])
    p.add_argument("--port", default=7860, type=int)
    p.add_argument("--share", action="store_true")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    adagam_net, _ = load_model(args.adagam_ckpt, device, args.which)
    base_net = load_model(args.base_ckpt, device, args.which)[0] if args.base_ckpt else None
    sample_pts, sample_lbl = _load_sample_bank(args.data)

    demo = create_app(adagam_net, base_net, sample_pts, sample_lbl, device)
    kw = dict(server_name="0.0.0.0", server_port=args.port, share=args.share)
    import inspect
    if "theme" in inspect.signature(demo.launch).parameters:   # Gradio 6 moved theme to launch()
        kw["theme"] = gr.themes.Soft()
    demo.launch(**kw)


if __name__ == "__main__":
    main()
