# AdaGAM: Adaptive Geometric Affine Modulation for Robust 3D Point Cloud Classification

**AdaGAM** is a lightweight, permutation-invariant, self-conditioned FiLM modulation layer for 3D point cloud networks. It replaces the static affine parameters $(\alpha, \beta)$ in PointMLP's local geometric grouper with input-conditioned residual modulation $(\Delta\alpha(g), \Delta\beta(g))$ computed from pooled neighborhood statistics, initialized at identity.

**▶ Live demo (runs in your browser): https://ankur777jinn.github.io/AdaGAM/** — PointMLP, AdaGAM and AdaGAM with its adaptation switched off, on official ModelNet-C test objects or your own point cloud.

## Key Features

- **Identity-Initialized Residual FiLM**: $\hat{f}_{j,k} = \alpha \odot (1 + \Delta\alpha(g_j)) \odot \tilde{f}_{j,k} + (\beta + \Delta\beta(g_j))$ with zero-initialized output projection (`+1.35%` params, `+0.11%` FLOPs at `r=8` on PointMLP).
- **Variable-$N$ Support**: Dynamic group count scaling (`N // reduce`) handles ModelNet-C point-dropping and point-adding corruptions.
- **Corruption-Type Linear Probe (`probe_corruption.py`)**: Evaluates whether AdaGAM's internal context vectors $g$ implicitly detect corruption types without corruption supervision.
- **Per-Point Adaptation Saliency (`saliency_map.py`, `plot_saliency.py`)**: Projects Stage-1 $|\Delta\alpha|$ magnitudes back onto 3D points via $k$-NN membership for interactive 3D visualization.
- **Live Inference Screening (`demo_app.py`)**: Gradio + Plotly 3D web app with a live counterfactual `AdaGAM ON / OFF (Δα=0, Δβ=0)` toggle, corruption/severity controls, and stage-wise $|\Delta\alpha|$ telemetry.

## Setup (RTX A6000 / CUDA)

```bash
git clone https://github.com/ankur777jinn/AdaGAM.git
cd AdaGAM
conda create -n adagam python=3.10 -y && conda activate adagam
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# Optional (recommended for fast GPU FPS; falls back to pure-PyTorch FPS if omitted):
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"

cd classification_ModelNet40
python smoke_test.py
```

**ModelNet-C Data**: Download from [jiawei-ren/ModelNet-C](https://github.com/jiawei-ren/ModelNet-C) and place `.h5` files in `classification_ModelNet40/data/modelnet_c/`.

## Running Experiments

```bash
cd classification_ModelNet40

# 1. Time one epoch
CUDA_VISIBLE_DEVICES=0 python main.py --model pointMLP --epoch 1 --msg speedtest --gam adagam --reduction 8

# 2. Train & evaluate the 4-run matrix (base, adagam, base_aug, adagam_aug)
GPU=0 R=8 bash run_experiments.sh train
GPU=0 bash run_experiments.sh eval
```

## Analysis & Live Demo

```bash
# Corruption-Type Linear Probe
CUDA_VISIBLE_DEVICES=0 python probe_corruption.py --ckpt checkpoints/pointMLP-adagam-42 --data data/modelnet_c

# Per-Point Adaptation Saliency (3D Plotly HTML + PNG)
CUDA_VISIBLE_DEVICES=0 python saliency_map.py --ckpt checkpoints/pointMLP-adagam-42 --data data/modelnet_c --out saliency_output/
python plot_saliency.py --input saliency_output/ --out saliency_figures/

# Feature t-SNE (Static vs AdaGAM)
CUDA_VISIBLE_DEVICES=0 python dump_features.py --ckpt checkpoints/pointMLP-base-42 --data data/modelnet_c
CUDA_VISIBLE_DEVICES=0 python dump_features.py --ckpt checkpoints/pointMLP-adagam-42 --data data/modelnet_c
python plot_tsne.py --base checkpoints/pointMLP-base-42/features.npz --adagam checkpoints/pointMLP-adagam-42/features.npz

# Live Inference Screening Web App (official ModelNet-C files, deterministic)
# 1) find objects where switching the adaptation off flips the prediction -> showcase.md
CUDA_VISIBLE_DEVICES=0 python find_showcase.py --base checkpoints/pointMLP-base-42 --adagam checkpoints/pointMLP-adagam-42 --data data/modelnet_c
# 2) launch; open http://localhost:7860 (from a laptop: ssh -L 7860:localhost:7860 user@server)
python demo_app.py --adagam_ckpt checkpoints/pointMLP-adagam-42 --base_ckpt checkpoints/pointMLP-base-42 --data data/modelnet_c

# Static in-browser demo (GitHub Pages): exports both models to ONNX (checked against PyTorch),
# packs demo objects and copies web/ -> web_build/ ; publish web_build/ on the gh-pages branch
python build_web.py --data data/modelnet_c --base checkpoints/pointMLP-base-42 --adagam checkpoints/pointMLP-adagam-42 --out web_build
```

## Explainability: where each model looks, and whether AdaGAM repairs feature drift

Both analyses run on trained checkpoints only (no extra training) and take well under an hour on one GPU.
Check them first on CPU with fake data: `python smoke_test_xai.py`.

```bash
cd classification_ModelNet40

# 1. Point saliency, PointMLP vs AdaGAM, on the same ModelNet-C objects
#    (Zheng et al. ICCV 2019 saliency; focus score; attention on injected points; faithfulness drop curves)
CUDA_VISIBLE_DEVICES=0 python xai_saliency.py --base checkpoints/pointMLP-base-42 \
    --adagam checkpoints/pointMLP-adagam-42 --data data/modelnet_c --out xai_out/saliency
#  -> fig_focus_shift.png, fig_outlier_attention.png, fig_drop_curves.png, saliency3d_obj*.html

# 2. Feature drift per stage: linear CKA(clean, corrupted) for PointMLP, AdaGAM,
#    AdaGAM with adaptation switched off, and PointMLP with BatchNorm re-estimated (Schneider et al. 2020)
CUDA_VISIBLE_DEVICES=0 python cka_drift.py --base checkpoints/pointMLP-base-42 \
    --adagam checkpoints/pointMLP-adagam-42 --data data/modelnet_c --out xai_out/cka
#  -> fig_cka_stages.png, fig_cka_gain_heatmap.png, fig_accuracy.png, cka_results.json
```

Read the drop curves before trusting any saliency figure: removing the most salient points must hurt accuracy
far more than removing random points. CKA is descriptive (similarity of representations), not proof of a
change in function; BN-adapt needs a whole batch of one corruption, AdaGAM adapts per object.

## References

- **PointMLP**: Ma, X., Qin, C., You, H., Ran, H., & Fu, Y. (2022). *Rethinking Network Design and Local Geometry in Point Cloud: A Simple Residual MLP Framework*. ICLR 2022.
- **ModelNet-C**: Ren, J., Pan, L., & Liu, Z. (2022). *Benchmarking and Analyzing Point Cloud Classification under Corruptions*. ICML 2022.
- **FiLM**: Perez, E., Strub, F., de Vries, H., Dumoulin, V., & Courville, A. (2018). *FiLM: Visual Reasoning with a General Conditioning Layer*. AAAI 2018.
