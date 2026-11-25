"""Live demo: PointMLP vs PointMLP + AdaGAM on clean and corrupted point clouds (Gradio + Plotly).

    python demo_app.py                                                     # no checkpoint: demo weights
    python demo_app.py --adagam_ckpt checkpoints/pointMLPElite-adagam-42 \
                       --base_ckpt   checkpoints/pointMLPElite-base-42 --data data/modelnet_c

Inputs: a procedural shape, a ModelNet-C test object (if --data has clean.h5), or an uploaded cloud
(.npy .npz .xyz .txt .csv .ply .obj); a corruption (Clean + 7 ModelNet-C corruptions) and severity 1-5.
Corruptions are seeded from (input, corruption, severity, seed) only, so toggling the adaptation never
changes the cloud. With --data, the official ModelNet-C file for that corruption is used for test objects.
"""
import argparse
import hashlib
import os
from typing import Optional

import gradio as gr
import h5py
import numpy as np
import plotly.graph_objects as go

from corruptions import CORRUPTIONS, apply_corruption
from data import SHAPE_NAMES
from inference_engine import InferenceEngine
from pc_io import EXTS, UploadError, load_point_cloud
from shapes import make_shape, shape_names

SOURCES = ["Procedural shape", "ModelNet-C test object", "Upload"]
CORR_CHOICES = ["clean"] + CORRUPTIONS
COLORSCALE = [[0.0, "#cce3fd"], [0.2, "#9dc8f7"], [0.4, "#6aabf1"], [0.6, "#338ced"], [0.8, "#1b64be"], [1.0, "#0e3d7e"]]


class DemoState:
    def __init__(self, engine: InferenceEngine, data_dir: Optional[str] = None):
        self.engine = engine
        self.data_dir = data_dir if data_dir and os.path.exists(os.path.join(data_dir, "clean.h5")) else None
        self.n_test = 0
        if self.data_dir:
            with h5py.File(os.path.join(self.data_dir, "clean.h5"), "r") as f:
                self.n_test = int(f["data"].shape[0])

    def _h5(self, split, idx):
        with h5py.File(os.path.join(self.data_dir, f"{split}.h5"), "r") as f:
            return f["data"][int(idx)].astype(np.float32), int(np.asarray(f["label"][int(idx)]).reshape(-1)[0])

    def cloud(self, source, shape, sample_idx, upload, corruption, severity, seed):
        """Returns (points [N, 3], true label or None, description)."""
        seed = int(seed or 0)
        lvl = int(severity) - 1
        if not 0 <= lvl <= 4:
            raise ValueError("severity must be 1..5")
        key = f"{source}|{shape}|{sample_idx}|{upload}|{corruption}|{lvl}|{seed}"
        rng = np.random.default_rng(int(hashlib.sha1(key.encode()).hexdigest()[:8], 16))
        if source == "ModelNet-C test object":
            if not self.data_dir:
                raise UploadError("no ModelNet-C data: start the app with --data data/modelnet_c")
            idx = int(np.clip(int(sample_idx or 0), 0, self.n_test - 1))
            split = "clean" if corruption == "clean" else f"{corruption}_{lvl}"
            pts, label = self._h5(split, idx)
            return pts, SHAPE_NAMES[label], f"ModelNet-C `{split}.h5` object {idx} (official file)"
        if source == "Upload":
            pts = load_point_cloud(upload, 1024, seed)
            label, desc = None, f"uploaded file `{os.path.basename(upload)}`"
        else:
            if shape not in shape_names():
                raise UploadError(f"pick a shape from {shape_names()}")
            pts = make_shape(shape, 1024, seed)
            label, desc = shape, f"procedural `{shape}` (seed {seed})"
        if corruption != "clean":
            pts = apply_corruption(pts, corruption, lvl, rng)
            desc += f" + `{corruption}` severity {lvl + 1} (seeded)"
        return pts.astype(np.float32), label, desc


def make_plot(pts, sal, title, adapt):
    cmax = float(np.percentile(sal, 99)) if adapt and sal.max() > 0 else 1.0
    fig = go.Figure(go.Scatter3d(
        x=pts[:, 0], y=pts[:, 2], z=pts[:, 1], mode="markers",
        marker=dict(size=2.6, color=sal if adapt else np.zeros(len(pts)), colorscale=COLORSCALE, cmin=0,
                    cmax=max(cmax, 1e-6), colorbar=dict(title="|Δα|", thickness=12)),
        hovertemplate="|Δα| = %{marker.color:.4f}<extra></extra>"))
    axis = dict(showticklabels=False, title="", showgrid=False, zeroline=False, showbackground=False)
    fig.update_layout(title=dict(text=title, x=0.01, font=dict(size=13)), height=520,
                      margin=dict(l=0, r=0, t=40, b=0),
                      scene=dict(xaxis=axis, yaxis=axis, zaxis=axis, aspectmode="data"))
    return fig


def run(state: DemoState, source, shape, sample_idx, upload, corruption, severity, adapt, seed):
    try:
        pts, label, desc = state.cloud(source, shape, sample_idx, upload, corruption, severity, seed)
        r = state.engine.predict(pts, adapt=bool(adapt))
    except (UploadError, ValueError) as e:
        return None, {}, {}, f"⚠️ **{e}**"
    a_top = max(r["adagam_top5"], key=r["adagam_top5"].get)
    s_top = max(r["static_top5"], key=r["static_top5"].get)
    title = (f"AdaGAM {'ON' if adapt else 'OFF (counterfactual: Δα = Δβ = 0)'} · colour = stage-1 |Δα|"
             + (f" · true: {label}" if label else ""))
    fig = make_plot(pts, r["saliency"], title, adapt)
    stages = " · ".join(f"S{i + 1} {v:.4f}" for i, v in enumerate(r["stage_dalpha"]))
    tele = "\n".join([
        f"**Input:** {desc} · **{r['n_points']} points**",
        f"**Prediction:** static → `{s_top}` · AdaGAM → `{a_top}`" + (f" · true → `{label}`" if label else ""),
        f"**Mean |Δα| per stage:** {stages}",
        f"**Latency (batch 1):** static {r['ms_static']:.1f} ms · AdaGAM {r['ms_adagam']:.1f} ms"
        + (" (cached)" if r["cached"] else ""),
    ])
    return fig, r["static_top5"], r["adagam_top5"], tele


def banner(engine: InferenceEngine) -> str:
    lines = [f"### AdaGAM-3D live demo", engine.describe()]
    if engine.demo_weights:
        lines.append("> ⚠️ **Demo weights (untrained).** No checkpoint was given, so the predictions and |Δα| values are "
                     "random and mean nothing. Start with `--adagam_ckpt <run>` (and `--base_ckpt <run>`) for real results.")
    if engine.base_is_counterfactual and not engine.demo_weights:
        lines.append("> The *static* column is the AdaGAM model with its adaptation removed (no `--base_ckpt` given), "
                     "not a separately trained PointMLP.")
    return "\n\n".join(lines)


def build_app(state: DemoState) -> gr.Blocks:
    sources = SOURCES if state.data_dir else [s for s in SOURCES if s != "ModelNet-C test object"]
    with gr.Blocks(title="AdaGAM-3D demo") as app:
        gr.Markdown(banner(state.engine))
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                source = gr.Radio(sources, value=sources[0], label="Input")
                shape = gr.Dropdown(shape_names(), value="chair", label="Procedural shape")
                sample_idx = gr.Number(value=0, precision=0, label=f"ModelNet-C test index (0..{max(state.n_test - 1, 0)})",
                                       visible=bool(state.data_dir))
                upload = gr.File(label="Upload " + " ".join(EXTS), file_types=list(EXTS), type="filepath")
                corruption = gr.Radio(CORR_CHOICES, value="clean", label="Corruption")
                severity = gr.Slider(1, 5, value=3, step=1, label="Severity")
                adapt = gr.Checkbox(value=True, label="AdaGAM adaptation ON (untick = counterfactual, same weights)")
                seed = gr.Number(value=0, precision=0, label="Seed (shape + corruption)")
                go_btn = gr.Button("Run", variant="primary")
            with gr.Column(scale=2):
                plot = gr.Plot(label="Point cloud")
                tele = gr.Markdown()
        with gr.Row():
            static_lbl = gr.Label(num_top_classes=5, label="PointMLP (static GAM) · top-5")
            ada_lbl = gr.Label(num_top_classes=5, label="PointMLP + AdaGAM · top-5")
        inputs = [source, shape, sample_idx, upload, corruption, severity, adapt, seed]
        outputs = [plot, static_lbl, ada_lbl, tele]

        def fn(*args):
            return run(state, *args)

        for comp in (source, shape, corruption, severity, adapt):
            comp.change(fn, inputs, outputs)
        sample_idx.submit(fn, inputs, outputs)
        seed.submit(fn, inputs, outputs)
        upload.upload(fn, inputs, outputs)
        go_btn.click(fn, inputs, outputs)
        app.load(fn, inputs, outputs)
    return app


def main(argv=None):
    p = argparse.ArgumentParser("AdaGAM-3D live demo")
    p.add_argument("--adagam_ckpt", default=None, help="AdaGAM run folder or .pth (omit for demo weights)")
    p.add_argument("--base_ckpt", default=None, help="static PointMLP run folder or .pth")
    p.add_argument("--data", default=os.path.join("data", "modelnet_c"), help="ModelNet-C folder (optional)")
    p.add_argument("--model", default="pointMLPElite", help="backbone for demo weights")
    p.add_argument("--device", default=None)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    a = p.parse_args(argv)
    engine = InferenceEngine(a.adagam_ckpt, a.base_ckpt, model=a.model, device=a.device)
    state = DemoState(engine, a.data)
    print(engine.describe())
    if engine.demo_weights:
        print("WARNING: running with untrained demo weights (no --adagam_ckpt)")
    build_app(state).launch(server_name=a.host, server_port=a.port, share=a.share)


if __name__ == "__main__":
    main()
