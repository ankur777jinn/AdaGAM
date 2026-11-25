---
name: novelty-to-live-demo
description: >-
  Research-verified novelty framing and live inference screening frontend workflow
  for ML projects. Use when asked to check novelty, propose presentable additions,
  and design/build an interactive live inference demo.
---

# Novelty-to-Live-Demo Workflow

1. **Web Research & Novelty Audit**:
   - Search recent benchmarks, leaderboards, and prior work on the target mechanism.
   - Explicitly separate borrowed primitives (e.g., FiLM) from the project's specific contribution.
   - Identify potential reviewer objections and design an experiment or metric to answer them.

2. **Two Presentable Novel Additions**:
   - **Interpretability / Probing**: Prove the module learns meaningful representations (e.g., linear probe on internal context vectors).
   - **Spatial / Visual Attribution**: Project internal modulation/attention magnitudes back onto the input (e.g., per-point 3D saliency).

3. **Live Inference Screening Frontend (Gradio + Plotly)**:
   - Include sample selector + custom file upload + live perturbation/severity controls.
   - Provide a **counterfactual/ablation toggle** (e.g., Adaptation ON vs. OFF on the same weights) alongside a baseline comparison.
   - Render interactive 3D/2D Plotly visuals, top-k class confidence bars, and per-stage internal telemetry.
   - Keep the UI runnable on both GPU and CPU fallback for live presentations.
