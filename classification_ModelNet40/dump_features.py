"""
Extract global 512-d features from a trained checkpoint on ModelNet-C for t-SNE.

Usage:
    python dump_features.py --ckpt checkpoints/pointMLP-base-42 --data data/modelnet_c
    python dump_features.py --ckpt checkpoints/pointMLP-adagam-42 --data data/modelnet_c
"""

import argparse
import os

import h5py
import numpy as np
import torch

from eval_modelnet_c import CORRUPTIONS, load_model


@torch.no_grad()
def extract_global_features(net, data_np, device, bs):
    feats = []
    for i in range(0, len(data_np), bs):
        x = torch.from_numpy(data_np[i:i + bs]).to(device).permute(0, 2, 1)
        _, feat = net(x, return_feat=True)
        feats.append(feat.cpu().numpy())
    return np.concatenate(feats, axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="checkpoint directory")
    p.add_argument("--data", default="data/modelnet_c", help="ModelNet-C directory")
    p.add_argument("--which", default="last", choices=["last", "best"])
    p.add_argument("--severity", default=4, type=int, help="severity 0..4 for corrupted splits")
    p.add_argument("--n_per_split", default=300, type=int, help="samples per split (0 = all)")
    p.add_argument("--bs", default=32, type=int)
    p.add_argument("--seed", default=42, type=int)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net, train_args = load_model(args.ckpt, device, args.which)

    with h5py.File(os.path.join(args.data, "clean.h5"), "r") as f:
        n_total = len(f["data"])
    rng = np.random.RandomState(args.seed)
    idx = (
        np.sort(rng.choice(n_total, size=min(args.n_per_split, n_total), replace=False))
        if args.n_per_split > 0 else np.arange(n_total)
    )

    all_feats, all_corr, all_labels = [], [], []
    splits = [("clean", "clean.h5")] + [(c, f"{c}_{args.severity}.h5") for c in CORRUPTIONS]
    for cname, fname in splits:
        with h5py.File(os.path.join(args.data, fname), "r") as f:
            pts = f["data"][idx].astype("float32")
            lbl = f["label"][idx].astype("int64").reshape(-1)
        feat = extract_global_features(net, pts, device, args.bs)
        all_feats.append(feat)
        all_corr.extend([cname] * len(feat))
        all_labels.append(lbl)

    out_path = os.path.join(args.ckpt, "features.npz")
    np.savez(
        out_path,
        features=np.concatenate(all_feats, axis=0),
        corruptions=np.array(all_corr),
        labels=np.concatenate(all_labels, axis=0),
        gam=train_args.gam,
        severity=args.severity,
    )
    print(f"Saved {out_path} ({len(all_corr)} samples)")


if __name__ == "__main__":
    main()
