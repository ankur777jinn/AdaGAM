"""Shared helpers for the two explainability analyses (xai_saliency.py, cka_drift.py)."""

import os

import h5py
import numpy as np
import torch

CORRUPTIONS = ["scale", "jitter", "rotate", "dropout_global", "dropout_local", "add_global", "add_local"]
SHORT = {"scale": "Scale", "jitter": "Jitter", "rotate": "Rotate", "dropout_global": "Drop-G",
         "dropout_local": "Drop-L", "add_global": "Add-G", "add_local": "Add-L"}

# One colour per model, used in every figure (validated categorical order, dataviz reference palette).
COLORS = {"PointMLP": "#2a78d6", "AdaGAM": "#eb6834", "AdaGAM (adaptation off)": "#1baf7a",
          "PointMLP + BN-adapt": "#eda100"}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
# Diverging pair: red (worse) <-> neutral gray <-> blue (better)
DIVERGING = ["#b8312f", "#e34948", "#f0a3a2", "#f0efec", "#9ec5f4", "#3987e5", "#1c5cab"]
# Sequential blue ramp, light -> dark (point saliency)
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def load_split(data_dir, corruption, severity, idx=None):
    """corruption='clean' or a ModelNet-C name; severity 0..4 (file suffix)."""
    name = "clean.h5" if corruption == "clean" else f"{corruption}_{severity}.h5"
    with h5py.File(os.path.join(data_dir, name), "r") as f:
        data = f["data"][:].astype("float32")
        label = f["label"][:].astype("int64").reshape(-1)
    if idx is not None:
        data, label = data[idx], label[idx]
    return data, label


def adagam_modules(net):
    return [m for m in net.modules() if m.__class__.__name__ == "AdaGAM"]


def set_adaptation(net, on: bool):
    for m in adagam_modules(net):
        m.disable_adaptation = not on


def seeded(seed):
    """Same FPS start points for every model/variant, so differences come from the models."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mpl_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold", "lines.linewidth": 2, "legend.frameon": False,
    })


def diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("adagam_div", DIVERGING)


def sequential_plotly():
    n = len(SEQUENTIAL) - 1
    return [[i / n, c] for i, c in enumerate(SEQUENTIAL)]
