# AdaGAM: Adaptive Geometric Affine Modulation for Robust 3D Point Cloud Classification

**AdaGAM** is a lightweight, permutation-invariant, self-conditioned FiLM modulation layer for 3D point cloud networks. It replaces the static affine parameters $(\alpha, \beta)$ in PointMLP's local geometric grouper with input-conditioned residual modulation $(\Delta\alpha(g), \Delta\beta(g))$ computed from pooled neighborhood statistics, initialized at identity.

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

# Live Inference Screening Web App
python demo_app.py --adagam_ckpt checkpoints/pointMLP-adagam-42 --base_ckpt checkpoints/pointMLP-base-42 --data data/modelnet_c
```

## References

- **PointMLP**: Ma, X., Qin, C., You, H., Ran, H., & Fu, Y. (2022). *Rethinking Network Design and Local Geometry in Point Cloud: A Simple Residual MLP Framework*. ICLR 2022.
- **ModelNet-C**: Ren, J., Pan, L., & Liu, Z. (2022). *Benchmarking and Analyzing Point Cloud Classification under Corruptions*. ICML 2022.
- **FiLM**: Perez, E., Strub, F., de Vries, H., Dumoulin, V., & Courville, A. (2018). *FiLM: Visual Reasoning with a General Conditioning Layer*. AAAI 2018.
