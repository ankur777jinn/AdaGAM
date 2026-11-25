"""
Per-Point Adaptation Saliency Map Extraction (Novelty #2).

Computes per-neighborhood |Delta alpha| at Stage 1 (highest spatial resolution)
and projects those group-level modulation magnitudes back to individual 3D points
via k-NN neighborhood membership.

Usage:
    python saliency_map.py --ckpt checkpoints/pointMLP-adagam-42 \
        --data data/modelnet_c --out saliency_output/ \
        --n_samples 8 --corruptions clean,jitter,dropout_local,add_global --severity 4
"""

import argparse
import os

import h5py
import numpy as np
import torch

from eval_modelnet_c import load_model


@torch.no_grad()
def compute_point_saliency(net, pts_np, device):
    """
    pts_np: [B, N, 3] numpy array
    Returns:
        saliency: [B, N] numpy array of per-point |Delta alpha| (max over containing groups)
        preds:    [B] predicted class indices
    """
    adagam_mods = [m for m in net.modules() if m.__class__.__name__ == "AdaGAM"]
    groupers = [m for m in net.modules() if m.__class__.__name__ == "LocalGrouper"]
    assert len(adagam_mods) > 0, "Saliency requires an AdaGAM checkpoint (gam='adagam')."

    adagam_mods[0].record_stats = True
    groupers[0]._store_idx = True

    x = torch.from_numpy(pts_np).to(device).permute(0, 2, 1)
    logits = net(x)
    preds = logits.argmax(dim=1).cpu().numpy()

    # dalpha_grp: [B, S], knn_idx: [B, S, K]
    dalpha_grp = adagam_mods[0].last_dalpha_per_group
    knn_idx = groupers[0]._last_knn_idx

    adagam_mods[0].record_stats = False
    groupers[0]._store_idx = False

    B, N, _ = pts_np.shape
    _, S, K = knn_idx.shape
    saliency = torch.zeros((B, N), device=device, dtype=dalpha_grp.dtype)
    counts = torch.zeros((B, N), device=device, dtype=dalpha_grp.dtype)

    flat_idx = knn_idx.reshape(B, S * K)
    flat_val = dalpha_grp.unsqueeze(-1).expand(B, S, K).reshape(B, S * K)

    saliency.scatter_add_(1, flat_idx, flat_val)
    counts.scatter_add_(1, flat_idx, torch.ones_like(flat_val))
    saliency = saliency / torch.clamp(counts, min=1.0)

    return saliency.cpu().numpy(), preds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="AdaGAM checkpoint directory")
    p.add_argument("--data", default="data/modelnet_c", help="ModelNet-C directory")
    p.add_argument("--out", default="saliency_output", help="output directory for .npz files")
    p.add_argument("--which", default="last", choices=["last", "best"])
    p.add_argument("--corruptions", default="clean,jitter,dropout_local,add_global")
    p.add_argument("--severity", default=4, type=int, help="severity 0..4 for corrupted splits")
    p.add_argument("--n_samples", default=8, type=int, help="number of test objects to extract")
    p.add_argument("--seed", default=42, type=int)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net, train_args = load_model(args.ckpt, device, args.which)
    assert train_args.gam == "adagam", f"Expected gam='adagam', got {train_args.gam}"

    with h5py.File(os.path.join(args.data, "clean.h5"), "r") as f:
        n_total = len(f["data"])
    rng = np.random.RandomState(args.seed)
    sample_indices = rng.choice(n_total, size=min(args.n_samples, n_total), replace=False)

    corr_list = [c.strip() for c in args.corruptions.split(",") if c.strip()]
    for cname in corr_list:
        fname = "clean.h5" if cname == "clean" else f"{cname}_{args.severity}.h5"
        fpath = os.path.join(args.data, fname)
        with h5py.File(fpath, "r") as f:
            pts = f["data"][sample_indices].astype("float32")
            labels = f["label"][sample_indices].astype("int64").reshape(-1)

        saliency, preds = compute_point_saliency(net, pts, device)
        out_path = os.path.join(args.out, f"{cname}_sev{args.severity}.npz" if cname != "clean" else "clean.npz")
        np.savez(
            out_path,
            points=pts,
            saliency=saliency,
            labels=labels,
            preds=preds,
            sample_indices=sample_indices,
            corruption=cname,
            severity=0 if cname == "clean" else args.severity,
        )
        print(f"Saved {out_path} | mean |d_alpha|={saliency.mean():.4f} | max={saliency.max():.4f}")


if __name__ == "__main__":
    main()
