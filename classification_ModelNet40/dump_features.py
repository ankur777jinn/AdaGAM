"""Dump global features (the max-pooled vector fed to the classifier) on ModelNet-C splits.

Feature size: 1024-d for pointMLP, 256-d for pointMLPElite (read from the model, not hard-coded).
One file per split, so different corruptions/severities never overwrite each other:
    <out_dir>/features_<split>.npz  with  feats [n, d], labels [n], preds [n], split, run, gam

    python dump_features.py --ckpt checkpoints/pointMLPElite-adagam-42 --data data/modelnet_c --splits clean,add_local_2
"""
import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from data import load_h5
from eval_modelnet_c import load_model


@torch.no_grad()
def dump(net, path, device, batch_size=32, max_samples=0):
    data, label = load_h5(path)
    if max_samples:
        data, label = data[:max_samples], label[:max_samples]
    feats, preds = [], []
    for (x,) in DataLoader(TensorDataset(torch.from_numpy(data)), batch_size=batch_size):
        logits, f = net(x.to(device).permute(0, 2, 1), return_feat=True)
        feats.append(f.float().cpu().numpy())
        preds.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(feats), label, np.concatenate(preds)


def main(argv=None):
    p = argparse.ArgumentParser("Dump global features")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", default=os.path.join("data", "modelnet_c"))
    p.add_argument("--splits", default="clean,add_local_2", help="comma list of ModelNet-C file stems")
    p.add_argument("--which", default="last", choices=["last", "best"])
    p.add_argument("--out_dir", default=None, help="default: <ckpt>/features")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--device", default=None)
    a = p.parse_args(argv)
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    net, args, _ = load_model(a.ckpt, a.which, device)
    out_dir = a.out_dir or os.path.join(a.ckpt, "features")
    os.makedirs(out_dir, exist_ok=True)
    for split in [s.strip() for s in a.splits.split(",") if s.strip()]:
        feats, labels, preds = dump(net, os.path.join(a.data, f"{split}.h5"), device, a.batch_size, a.max_samples)
        path = os.path.join(out_dir, f"features_{split}.npz")
        np.savez_compressed(path, feats=feats, labels=labels, preds=preds, split=split,
                            run=os.path.basename(os.path.normpath(a.ckpt)), gam=args.get("gam", ""))
        print(f"  {split:18s} -> {path}  feats {feats.shape} (dim = {net.feat_dim}) acc {np.mean(preds == labels):.3f}")


if __name__ == "__main__":
    main()
