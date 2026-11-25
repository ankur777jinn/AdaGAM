"""
Corruption-Type Linear Probe on AdaGAM Context Vectors (Novelty #1).

Freezes a trained checkpoint, extracts the permutation-invariant context vectors g
from each LocalGrouper stage across clean and ModelNet-C corrupted point clouds,
and trains a linear classifier (LogisticRegression) to predict which of the 8
corruption types (clean + 7 corruptions) was applied.

For static PointMLP baselines (gam="static"), extracts the same [AvgPool_K; MaxPool_K]
context of the normalized patch just before the static affine parameters.

Usage:
    python probe_corruption.py --ckpt checkpoints/pointMLP-adagam-42 --data data/modelnet_c
    python probe_corruption.py --ckpt checkpoints/pointMLP-base-42 --data data/modelnet_c
"""

import argparse
import json
import os

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

from eval_modelnet_c import CORRUPTIONS, load_model

CLASSES = ["clean"] + CORRUPTIONS


@torch.no_grad()
def extract_context_features(net, gam_mode, data_np, device, bs):
    """Run forward pass and return [N, sum_d] concatenated pooled context vectors
    and a list of per-stage [N, d_s] arrays."""
    adagam_mods = [m for m in net.modules() if m.__class__.__name__ == "AdaGAM"]
    groupers = [m for m in net.modules() if m.__class__.__name__ == "LocalGrouper"]

    if gam_mode == "adagam":
        for m in adagam_mods:
            m.record_stats = True
    else:
        for g in groupers:
            g._store_pre_affine = True

    stage_batches = [[] for _ in range(len(groupers))]
    for i in range(0, len(data_np), bs):
        x = torch.from_numpy(data_np[i:i + bs]).to(device).permute(0, 2, 1)
        _ = net(x)
        if gam_mode == "adagam":
            for s, m in enumerate(adagam_mods):
                # m.last_context: [B, S, in_dim] -> pool over S -> [B, in_dim]
                stage_batches[s].append(m.last_context.mean(dim=1).cpu().numpy())
        else:
            for s, g in enumerate(groupers):
                stage_batches[s].append(g._last_pre_affine_ctx.mean(dim=1).cpu().numpy())

    for m in adagam_mods:
        m.record_stats = False
    for g in groupers:
        g._store_pre_affine = False

    per_stage = [np.concatenate(b, axis=0) for b in stage_batches]
    concat = np.concatenate(per_stage, axis=1)
    return concat, per_stage


def load_balanced_splits(data_dir, samples_per_class, seed):
    rng = np.random.RandomState(seed)
    all_x, all_y = [], []

    def _sample_file(path, n_take):
        with h5py.File(path, "r") as f:
            arr = f["data"][:].astype("float32")
        if 0 < n_take < len(arr):
            idx = rng.choice(len(arr), size=n_take, replace=False)
            arr = arr[idx]
        return arr

    clean_path = os.path.join(data_dir, "clean.h5")
    all_x.append(_sample_file(clean_path, samples_per_class))
    all_y.append(np.zeros(len(all_x[-1]), dtype=np.int64))

    per_sev = (samples_per_class // 5) if samples_per_class > 0 else 0
    for c_idx, cname in enumerate(CORRUPTIONS, start=1):
        for sev in range(5):
            path = os.path.join(data_dir, f"{cname}_{sev}.h5")
            arr = _sample_file(path, per_sev)
            all_x.append(arr)
            all_y.append(np.full(len(arr), c_idx, dtype=np.int64))

    return all_x, all_y


def fit_and_eval_probe(X, y, seed):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    acc = float(accuracy_score(y_te, preds))
    cm = confusion_matrix(y_te, preds, labels=list(range(len(CLASSES))))
    per_class = {
        CLASSES[i]: float(cm[i, i] / max(1, cm[i].sum()))
        for i in range(len(CLASSES))
    }
    return acc, per_class, cm


def plot_confusion(cm, acc, gam_mode, out_path):
    cm_norm = cm.astype(np.float64) / np.maximum(1, cm.sum(axis=1, keepdims=True))
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(CLASSES, fontsize=9)
    ax.set_xlabel("Predicted Corruption")
    ax.set_ylabel("True Corruption")
    ax.set_title(f"Corruption Linear Probe ({gam_mode}) — OA: {acc * 100:.1f}%")

    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            val = cm_norm[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="checkpoint directory")
    p.add_argument("--data", default="data/modelnet_c", help="ModelNet-C directory")
    p.add_argument("--which", default="last", choices=["last", "best"])
    p.add_argument("--bs", default=32, type=int)
    p.add_argument("--samples_per_class", default=500, type=int,
                   help="samples per corruption class (0 = use all)")
    p.add_argument("--seed", default=42, type=int)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net, train_args = load_model(args.ckpt, device, args.which)
    for param in net.parameters():
        param.requires_grad = False

    print(f"Loaded {args.ckpt} (gam={train_args.gam}) on {device}")
    chunks_x, chunks_y = load_balanced_splits(args.data, args.samples_per_class, args.seed)

    feats_concat, feats_stages, labels = [], [[] for _ in range(4)], []
    for x_arr, y_arr in zip(chunks_x, chunks_y):
        c, per_s = extract_context_features(net, train_args.gam, x_arr, device, args.bs)
        feats_concat.append(c)
        for s_idx, s_arr in enumerate(per_s):
            feats_stages[s_idx].append(s_arr)
        labels.append(y_arr)

    X = np.concatenate(feats_concat, axis=0)
    y = np.concatenate(labels, axis=0)
    print(f"Extracted probe dataset: X={X.shape}, classes={len(CLASSES)}")

    acc, per_class, cm = fit_and_eval_probe(X, y, args.seed)
    stage_accs = []
    for s_idx, s_chunks in enumerate(feats_stages):
        Xs = np.concatenate(s_chunks, axis=0)
        s_acc, _, _ = fit_and_eval_probe(Xs, y, args.seed)
        stage_accs.append(s_acc)

    print(f"\nLinear Probe Accuracy (all stages): {acc * 100:.2f}% (chance: {100.0 / len(CLASSES):.2f}%)")
    print("Per-stage accuracy:", [round(a * 100, 2) for a in stage_accs])
    for k, v in per_class.items():
        print(f"  {k:16s}: {v * 100:.2f}%")

    out_json = os.path.join(args.ckpt, "probe_results.json")
    out_png = os.path.join(args.ckpt, "probe_confusion.png")
    with open(out_json, "w") as f:
        json.dump({
            "ckpt": args.ckpt,
            "gam": train_args.gam,
            "overall_acc": acc,
            "chance_acc": 1.0 / len(CLASSES),
            "per_stage_acc": stage_accs,
            "per_class_acc": per_class,
            "confusion_matrix": cm.tolist(),
            "classes": CLASSES,
        }, f, indent=2)
    plot_confusion(cm, acc, train_args.gam, out_png)
    print(f"Saved {out_json} and {out_png}")


if __name__ == "__main__":
    main()
