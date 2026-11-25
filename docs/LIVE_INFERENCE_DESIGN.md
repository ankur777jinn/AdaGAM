# Live inference demo — design

Branch: `live-inference` (kept apart from `main` so the experiment code stays minimal).

## Goal

Show, on one screen, what AdaGAM does to a corrupted point cloud: the static PointMLP and the AdaGAM model
side by side, *where* in the cloud AdaGAM adapts (stage-1 |Δα| as colour), and what happens when the
adaptation is switched off on the same weights.

## Files (`classification_ModelNet40/`)

| File | Role |
|---|---|
| `demo_app.py` | Gradio Blocks UI and request handler `run(...)` |
| `inference_engine.py` | Loads models (checkpoints or demo weights), thread-safe `predict(points, adapt)` with a small LRU cache |
| `shapes.py` | 10 procedural, correctly labelled shapes (airplane, bottle, bowl, chair, cone, cup, monitor, stairs, table, tent) |
| `pc_io.py` | Pure-NumPy upload parser: `.npy .npz .xyz .txt .csv .ply` (ASCII + binary) `.obj` (vertices, or surface samples for small meshes) |
| `tests/test_demo.py` | Starts the engine with no checkpoint and no data, drives every control, checks errors |

## Request flow

1. **Cloud.** `DemoState.cloud()` builds the input from the chosen source. The corruption RNG is seeded from
   `(source, shape/index/file, corruption, severity, seed)` only, so ticking the adaptation box never reshuffles the
   cloud. For ModelNet-C test objects (`--data` given) the official corrupted `.h5` file is used, not our re-implementation.
2. **Inference.** `InferenceEngine.predict()` runs the static model once and the AdaGAM model once
   (`return_aux=True` returns logits, per-stage |Δα| and the stage-1 kNN indices together). The counterfactual
   is `forward(x, adapt=False)`: an argument, not a module flag, so concurrent requests cannot interfere.
   A lock serialises GPU use; results are cached by (cloud bytes, adapt).
3. **Saliency.** `saliency_map.point_saliency()` spreads each stage-1 neighbourhood's |Δα| to its K points and
   averages per point (no backward pass).
4. **Display.** Plotly `Scatter3d` coloured by saliency (one-hue blue scale), two `gr.Label` top-5 panels,
   telemetry: input description, point count (variable for dropout/add corruptions), per-stage |Δα|, batch-1 latency.

## Modes

| Start command | Static column | AdaGAM column |
|---|---|---|
| `python demo_app.py` | AdaGAM backbone with adaptation removed | untrained demo weights (banner: predictions are meaningless) |
| `--adagam_ckpt RUN` | AdaGAM weights with adaptation removed (banner says so) | trained AdaGAM |
| `--adagam_ckpt RUN --base_ckpt RUN` | separately trained PointMLP | trained AdaGAM |

In demo-weights mode the AdaGAM `fc2` layers get small random weights (std 0.02) so the colour map is not flat;
this is only to exercise the interface and is stated in the banner.

## Errors

Every input problem becomes a `⚠️` message in the telemetry box (no stack trace): wrong extension, file > 50 MB,
object/pickled `.npy`, wrong shape, NaN/inf rows, fewer than 32 valid points, identical points, missing file,
ModelNet-C source without `--data`.

## Latency

Batch-1 latency is shown per click. On this repository's CPU test machine one click took ~0.3 s with demo weights;
the RTX 4050 number is TBD (measure with the app or `audit_overhead.py`).
