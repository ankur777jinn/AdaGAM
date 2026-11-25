"""Novelty #2 - single-pass per-point adaptation saliency.

For every input point we report how strongly AdaGAM changed the affine around it: the stage-1
|d_alpha| of each neighbourhood (mean over channels) is spread back to the points of that
neighbourhood through its kNN membership, and each point gets the MEAN over the neighbourhoods that
contain it. No backward pass is needed (unlike Grad-CAM-style saliency): it is read off the same
forward pass that classifies the cloud. Points that no stage-1 neighbourhood contains get 0 and
`covered = False`.

    python saliency_map.py --ckpt checkpoints/pointMLPElite-adagam-42 --data data/modelnet_c --out saliency_output/
Writes one saliency_<split>.npz per split (clean + each corruption at --severity).
"""
import argparse
import os

import h5py
import numpy as np
import torch

from corruptions import CORRUPTIONS, canonical
from eval_modelnet_c import load_model


def point_saliency(dalpha_group: torch.Tensor, knn_idx: torch.Tensor, n_points: int):
    """dalpha_group [B, S], knn_idx [B, S, K] (indices into the N input points) -> ([B, N], covered [B, N])."""
    B, S, K = knn_idx.shape
    vals = dalpha_group.unsqueeze(-1).expand(B, S, K).reshape(B, -1).float()
    idx = knn_idx.reshape(B, -1)
    total = torch.zeros(B, n_points, device=vals.device).scatter_add_(1, idx, vals)
    count = torch.zeros(B, n_points, device=vals.device).scatter_add_(1, idx, torch.ones_like(vals))
    covered = count > 0
    return torch.where(covered, total / count.clamp(min=1), torch.zeros_like(total)), covered


@torch.no_grad()
def run_split(net, path, sample_indices, device, batch_size=16):
    idx = np.sort(np.asarray(sample_indices))                     # h5py fancy indexing needs increasing order
    with h5py.File(path, "r") as f:
        pcs = f["data"][idx].astype("float32")
        labels = f["label"][idx].astype("int64").reshape(-1)
    sal, cov, preds, preds_cf = [], [], [], []
    for b in range(0, len(idx), batch_size):
        x = torch.from_numpy(pcs[b:b + batch_size]).to(device).permute(0, 2, 1)
        logits, _, aux = net(x, return_aux=True)
        s, c = point_saliency(aux[0]["dalpha_group"], aux[0]["knn_idx"], x.shape[2])
        sal.append(s.cpu().numpy())
        cov.append(c.cpu().numpy())
        preds.append(logits.argmax(1).cpu().numpy())
        preds_cf.append(net(x, adapt=False).argmax(1).cpu().numpy())   # counterfactual: same weights, d=0
    return {"points": pcs, "saliency": np.concatenate(sal), "covered": np.concatenate(cov),
            "labels": labels, "preds": np.concatenate(preds), "preds_counterfactual": np.concatenate(preds_cf),
            "sample_indices": idx}


def main(argv=None):
    p = argparse.ArgumentParser("Adaptation saliency (Novelty #2)")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", default=os.path.join("data", "modelnet_c"))
    p.add_argument("--which", default="last", choices=["last", "best"])
    p.add_argument("--corruptions", default=",".join(CORRUPTIONS))
    p.add_argument("--severity", type=int, default=2, help="0..4")
    p.add_argument("--n_samples", type=int, default=8)
    p.add_argument("--sample_indices", default=None, help="comma list of test indices (overrides --n_samples)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--out", default="saliency_output")
    p.add_argument("--device", default=None)
    a = p.parse_args(argv)
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    net, args, _ = load_model(a.ckpt, a.which, device)
    if args.get("gam") != "adagam":
        raise SystemExit("saliency needs an AdaGAM run (gam=adagam); a static run has no d_alpha")
    with h5py.File(os.path.join(a.data, "clean.h5"), "r") as f:
        n_total = f["data"].shape[0]
    if a.sample_indices:
        sample_indices = np.array([int(i) for i in a.sample_indices.split(",")])
    else:
        sample_indices = np.random.default_rng(a.seed).choice(n_total, size=min(a.n_samples, n_total), replace=False)
    sample_indices = np.sort(sample_indices)
    os.makedirs(a.out, exist_ok=True)
    splits = ["clean"] + [f"{canonical(c)}_{a.severity}" for c in a.corruptions.split(",") if c.strip()]
    for split in splits:
        res = run_split(net, os.path.join(a.data, f"{split}.h5"), sample_indices, device, a.batch_size)
        np.savez_compressed(os.path.join(a.out, f"saliency_{split}.npz"), split=split, run=args.get("msg", ""), **res)
        print(f"  {split:18s} mean|da|={res['saliency'].mean():.5f} covered={res['covered'].mean():.3f} "
              f"acc={np.mean(res['preds'] == res['labels']):.3f} acc(adapt off)={np.mean(res['preds_counterfactual'] == res['labels']):.3f}")
    print(f"==> wrote {len(splits)} files to {a.out}/  (samples {sample_indices.tolist()})")


if __name__ == "__main__":
    main()
