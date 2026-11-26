# AdaGAM-3D: Adaptive Geometric Affine Modulation for Robust Point Cloud Classification

DAC-301 Deep Learning course project · Ankur Dahiya, Gaurav kumar, Diptanshu

PointMLP (Ma et al., ICLR 2022) normalises every local neighbourhood and then applies a learned affine
`α·x + β`. After training, `α` and `β` are fixed numbers, whatever input arrives. **AdaGAM** makes them
input-dependent: a small MLP reads each neighbourhood's own statistics (mean and max pooling over its
K = 24 neighbours) and predicts a per-neighbourhood correction

```
out = α · (1 + Δα_j) · x̃ + (β + Δβ_j),     [Δα_j, Δβ_j] = W2 · LeakyReLU(W1 · [mean_K x̃_j ; max_K x̃_j])
```

`W2` starts at zero, so an untrained AdaGAM is *exactly* PointMLP (max difference 0.0), and
`adapt=False` switches the adaptation off on the same weights (a counterfactual for ablations and the demo).
We measure robustness on **ModelNet-C** (Ren et al., ICML 2022): 7 corruptions × 5 severities, mCE normalised to DGCNN.

## Results

| | Clean OA | mCE ↓ | RmCE ↓ | Unseen-corruption OA (`_aug` runs) |
|---|---|---|---|---|
| PointMLP-Elite (`base`, clean-trained) | TBD (needs GPU run) | TBD | TBD | – |
| PointMLP-Elite + AdaGAM (`adagam`, clean-trained) | TBD (needs GPU run) | TBD | TBD | – |
| `base_aug` (not leaderboard-comparable) | TBD | TBD | TBD | TBD |
| `adagam_aug` (not leaderboard-comparable) | TBD | TBD | TBD | TBD |

The full table is generated into `classification_ModelNet40/results.md` by `run_experiments.* eval`.
**Headline = clean-trained `base` vs clean-trained `adagam`, last checkpoint.**

### Verified overhead (`audit_overhead.py`, batch 1, 1,024 points, FLOPs = 2 × MACs)

| Model | r | Params | +Params | GFLOPs | +FLOPs |
|---|---|---|---|---|---|
| pointMLP | – | 13.2385M | – | 31.212 | – |
| pointMLP + AdaGAM | 4 | 13.5888M | +2.65% | 31.274 | +0.20% |
| pointMLP + AdaGAM | **8** | **13.4167M** | **+1.35%** | **31.245** | **+0.11%** |
| pointMLPElite | – | 0.7162M | – | 1.794 | – |
| pointMLPElite + AdaGAM | 8 | 0.7644M | +6.72% | 1.804 | +0.58% |

GPU latency: TBD (needs GPU run; `audit_overhead.py` labels it with the device name, e.g. "measured on RTX 4050 Laptop GPU").
Elite's larger relative overhead comes from the `max(16, C // r)` floor on its small channel counts.

## Repository

```
AdaGAM_v2/
├── README.md · LICENSE · requirements.txt · ANALYSIS.md
├── docs/RELATED_WORK.md                 related work + honest novelty statement
└── classification_ModelNet40/
    ├── models/adagam.py                 the AdaGAM modulator (zero-init residual FiLM)
    ├── models/pointmlp.py               PointMLP backbone, gam="static" | "adagam", fast torch FPS fallback
    ├── main.py                          training (AMP, auto batch size, resume, corruption-consistency training)
    ├── data.py · corruptions.py         ModelNet40 loader / downloader, ModelNet-C-style training corruptions
    ├── eval_modelnet_c.py               OA, CE/mCE, RCE/RmCE, mCE vs baseline, |Δα| per stage (resumable)
    ├── summarize.py                     all runs -> results.md / results.csv (with seen/unseen columns)
    ├── run_experiments.py|.sh|.ps1      the 4-run matrix, sequential, Windows + Linux
    ├── audit_overhead.py                params / FLOPs / latency
    ├── probe_corruption.py              Novelty #1: corruption-type linear probe
    ├── saliency_map.py · plot_saliency.py   Novelty #2: single-pass per-point adaptation saliency
    ├── dump_features.py · plot_tsne.py  global features + t-SNE
    ├── plot_figures.py · viz_style.py   OA-vs-severity, CE bars, |Δα| heatmap, corruption gallery
    ├── smoke_test.py                    CPU sanity checks (run first)
    ├── demo_app.py · inference_engine.py · shapes.py · pc_io.py   live demo (this branch)
    └── tests/                           fake data + end-to-end tests
```

The live demo (`demo_app.py`, Gradio) lives on the **`live-inference`** branch (this branch). See below.

## Setup

### Windows laptop (RTX 4050)

Use the interpreter that has the ML packages (bare `python` on this laptop is an MSYS2 Python with none):

```powershell
git clone https://github.com/ankur777jinn/AdaGAM-3D.git
cd AdaGAM-3D
$PY = "C:\Users\Ankur\AppData\Local\Programs\Python\Python310\python.exe"
& $PY -m pip install -r requirements.txt          # torch 2.5.1+cu118 is already installed (supports sm_89)
cd classification_ModelNet40
& $PY smoke_test.py                                # must end with "All smoke tests passed."
& $PY -c "import torch; print(torch.cuda.get_device_name(0))"
```

`pointnet2_ops` (CUDA farthest point sampling) is optional and hard to build on Windows. Without it the code
uses a batched pure-PyTorch FPS (one Python loop over the sampled points, vectorised over the batch).
It starts at point 0 in eval mode, so evaluation is deterministic.

### Linux / WSL2

```bash
conda create -n adagam python=3.10 -y && conda activate adagam
pip install torch --index-url https://download.pytorch.org/whl/cu124   # pick the index matching your CUDA
pip install -r requirements.txt
# optional, ~faster FPS: pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
cd classification_ModelNet40 && python smoke_test.py
```

### Data

* **ModelNet40** downloads on the first run (`python main.py --download_only`). The original Stanford link is
  dead, so `data.py` tries a Hugging Face mirror first. If both fail: download `modelnet40_ply_hdf5_2048.zip`
  yourself and unzip it into `classification_ModelNet40/data/`, or set `MODELNET40_URL`.
* **ModelNet-C** (Google Drive, from the official repo): `pip install gdown`, then
  `python -c "import data; data.download_modelnet_c()"`, or download
  <https://drive.google.com/file/d/1KE6MmXMtfu_mgxg4qLPdEwVD5As8B0rm/view> and unzip it so that
  `classification_ModelNet40/data/modelnet_c/clean.h5` exists (36 files: `clean`, `<corruption>_<0..4>`).
  ModelNet-C is **only** used for evaluation.

## Running the experiments (RTX 4050, ~6 GB)

Primary model: `pointMLPElite` (0.72M params). Full `pointMLP` (13.2M) is optional.

```powershell
cd classification_ModelNet40
# 1) time one epoch -> prints a recommended EPOCHS for an 8-hour budget (change with --budget_hours)
& $PY main.py --model pointMLPElite --gam adagam --reduction 8 --amp --epoch 1 --msg speedtest
# 2) train the 4 runs one after another (re-running the same command resumes)
.\run_experiments.ps1 train -Epochs <EPOCHS>
# 3) evaluate the LAST checkpoints on ModelNet-C and write results.md
.\run_experiments.ps1 eval -Epochs <EPOCHS>
```

Linux / Git-Bash equivalent: `GPU=0 EPOCHS=<n> R=8 bash run_experiments.sh train` then `... eval`
(`DRY=1` prints the commands only). `run_experiments.py` is the shared implementation.

| Run | Training | Role |
|---|---|---|
| `base` | clean ModelNet40 | PointMLP baseline — **headline** |
| `adagam` | clean ModelNet40 | AdaGAM — **headline** (protocol-compliant) |
| `base_aug` | clean + corrupted views, consistency loss | fair baseline for the next row |
| `adagam_aug` | clean + corrupted views, consistency loss | AdaGAM with a training signal |

The `_aug` runs train on `jitter, dropout_global, add_global, scale` only. ModelNet-C forbids training on
similar corruptions, so these runs are **not leaderboard-comparable**; only their *unseen* corruptions
(`dropout_local, add_local, rotate`) are a generalisation result. Optional ablations: `--ablations`
(no consistency loss, std pooling, mid-term per-patch normalisation via `--gam_norm patch`).

Useful `main.py` flags: `--amp` (fp16 autocast + GradScaler; `--amp_dtype bf16` also works on the 4050),
`--batch_size 32` (halved automatically if it does not fit), `--num_workers 0` (Windows default),
`--resume`, `--max_batches N` (debug), `--fps torch|cuda|auto`.

### After training

```powershell
& $PY audit_overhead.py --model pointMLPElite --reduction 8                 # GPU latency (labelled with the GPU name)
& $PY probe_corruption.py --ckpt checkpoints\pointMLPElite-adagam-42 --data data\modelnet_c
& $PY probe_corruption.py --ckpt checkpoints\pointMLPElite-base-42   --data data\modelnet_c
& $PY saliency_map.py --ckpt checkpoints\pointMLPElite-adagam-42 --data data\modelnet_c --out saliency_output
& $PY plot_saliency.py --input saliency_output --out saliency_figures
& $PY dump_features.py --ckpt checkpoints\pointMLPElite-base-42   --splits clean,add_local_2
& $PY dump_features.py --ckpt checkpoints\pointMLPElite-adagam-42 --splits clean,add_local_2
& $PY plot_tsne.py --static checkpoints\pointMLPElite-base-42 --adagam checkpoints\pointMLPElite-adagam-42 --corruption add_local_2 --out figures\tsne.png
& $PY plot_figures.py --root checkpoints --data data\modelnet_c --out figures
```

## Live demo (`live-inference` branch)

```powershell
cd classification_ModelNet40
& $PY demo_app.py                                   # no checkpoint needed: untrained demo weights + banner
& $PY demo_app.py --adagam_ckpt checkpoints\pointMLPElite-adagam-42 --base_ckpt checkpoints\pointMLPElite-base-42 --data data\modelnet_c
# open http://127.0.0.1:7860
```

Pick a procedural shape (10 labelled classes), a ModelNet-C test object (with `--data`, using the official
corrupted files) or upload a cloud (`.npy .npz .xyz .txt .csv .ply .obj`); choose a corruption and severity.
The 3D view is coloured by AdaGAM's stage-1 |Δα|; the checkbox switches the adaptation off on the same weights
(counterfactual) without changing the cloud. Design: `docs/LIVE_INFERENCE_DESIGN.md`. Tests:
`& $PY tests\test_demo.py`.

## Colab / Kaggle fallback (if the laptop is too slow)

```python
!git clone https://github.com/ankur777jinn/AdaGAM-3D.git
%cd AdaGAM-3D/classification_ModelNet40
!pip install -q h5py scikit-learn matplotlib plotly gdown
!python smoke_test.py
!python main.py --download_only
!python -c "import data; data.download_modelnet_c()"
!python main.py --model pointMLPElite --gam adagam --amp --epoch 1 --msg speedtest
!GPU=0 EPOCHS=100 bash run_experiments.sh train      # Colab sessions time out: re-run the same line to resume
!GPU=0 EPOCHS=100 bash run_experiments.sh eval
```

Save `checkpoints/` to Google Drive (or Kaggle output) between sessions; every run resumes from
`last_checkpoint.pth`. On Kaggle, set `WORKERS=2`.

## Notes on protocol

* Evaluate the **last** checkpoint (`--which last`, default). Upstream PointMLP picks the best epoch on the
  test set, which flatters results; `best_checkpoint.pth` is still saved for reference.
* One seed per run; treat OA differences below ~0.5% as within run-to-run noise.
* DGCNN reference accuracies in `eval_modelnet_c.py` were checked against the official ModelNet-C code
  (`modelnetc_utils/eval.py`) on 2026-11-25.

## References

* X. Ma et al., *Rethinking Network Design and Local Geometry in Point Cloud: A Simple Residual MLP Framework* (PointMLP), ICLR 2022. <https://github.com/ma-xu/pointMLP-pytorch>
* J. Ren, L. Pan, Z. Liu, *Benchmarking and Analyzing Point Cloud Classification under Corruptions* (ModelNet-C), ICML 2022. <https://arxiv.org/abs/2202.03377> · <https://github.com/jiawei-ren/ModelNet-C>
* E. Perez et al., *FiLM: Visual Reasoning with a General Conditioning Layer*, AAAI 2018. <https://arxiv.org/abs/1709.07871>
* See `docs/RELATED_WORK.md` for the full comparison table.

## License

MIT © 2026 Ankur Dahiya. The PointMLP backbone is re-implemented from the official design (Apache-2.0).
