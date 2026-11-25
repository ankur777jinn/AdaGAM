"""
Evaluate a trained checkpoint on ModelNet-C (Ren et al., ICML 2022).

Expected data layout (download from the ModelNet-C repo: github.com/jiawei-ren/ModelNet-C):
    data/modelnet_c/clean.h5
    data/modelnet_c/{scale,jitter,rotate,dropout_global,dropout_local,add_global,add_local}_{0..4}.h5

Usage:
    python eval_modelnet_c.py --ckpt checkpoints/<run_dir>
    python eval_modelnet_c.py --ckpt checkpoints/<adagam_run> --ref checkpoints/<baseline_run>/modelnet_c.json

Outputs <ckpt>/modelnet_c.json and <ckpt>/modelnet_c.csv.

Metrics
    OA_{c,l}  overall accuracy on corruption c, severity l
    CE_c      = mean_l (1 - OA_{c,l})
    mCE       = mean_c CE_c / CE_c^DGCNN            (official ModelNet-C definition, DGCNN = 1.0)
    RmCE      = mean_c (OA_clean - OA_c) / (OA_clean^DGCNN - OA_c^DGCNN)
    mCE_vs_ref= mean_c CE_c / CE_c^ref               (relative to your own baseline run, if --ref)
"""

import argparse
import json
import os
import time
from types import SimpleNamespace

import h5py
import numpy as np
import torch

import models

CORRUPTIONS = ["scale", "jitter", "rotate", "dropout_global", "dropout_local", "add_global", "add_local"]
# DGCNN reference OAs from the ModelNet-C paper / README (verify against the repo before quoting).
DGCNN_OA = {"clean": 0.926, "scale": 0.906, "jitter": 0.684, "rotate": 0.785,
            "dropout_global": 0.752, "dropout_local": 0.793, "add_global": 0.705, "add_local": 0.725}


def load_model(ckpt_dir, device, which="last"):
    with open(os.path.join(ckpt_dir, "args.json")) as f:
        a = SimpleNamespace(**json.load(f))
    gam_kwargs = dict(reduction=a.reduction, use_std=a.use_std, norm=a.gam_norm)
    net = models.__dict__[a.model](gam=a.gam, gam_kwargs=gam_kwargs)
    state = torch.load(os.path.join(ckpt_dir, f"{which}_checkpoint.pth"), map_location="cpu",
                       weights_only=False)
    state = state.get("net", state)
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    net.load_state_dict(state)
    return net.to(device).eval(), a


@torch.no_grad()
def evaluate_file(net, path, device, bs):
    with h5py.File(path, "r") as f:
        data = f["data"][:].astype("float32")
        label = f["label"][:].astype("int64").reshape(-1)
    adagam_mods = [m for m in net.modules() if m.__class__.__name__ == "AdaGAM"]
    for m in adagam_mods:
        m.record_stats = True
    correct, d_alpha = 0, []
    for i in range(0, len(data), bs):
        x = torch.from_numpy(data[i:i + bs]).to(device).permute(0, 2, 1)
        pred = net(x).argmax(1).cpu().numpy()
        correct += (pred == label[i:i + bs]).sum()
        if adagam_mods:
            d_alpha.append([m.last_stats["d_alpha_abs"] for m in adagam_mods])
    for m in adagam_mods:
        m.record_stats = False
    stats = np.mean(d_alpha, 0).tolist() if d_alpha else None
    return correct / len(data), data.shape[1], stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="checkpoint directory written by main.py")
    p.add_argument("--data", default="data/modelnet_c")
    p.add_argument("--ref", default=None, help="modelnet_c.json of a baseline run for relative mCE")
    p.add_argument("--bs", default=32, type=int)
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--which", default="last", choices=["last", "best"],
                   help="last = final epoch (honest); best = picked on the clean test set (upstream habit)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net, train_args = load_model(args.ckpt, device, args.which)

    res = {"run": args.ckpt, "gam": train_args.gam, "oa": {}, "n_points": {}, "adagam_dalpha": {}}
    t0 = time.time()
    oa_clean, n, st = evaluate_file(net, os.path.join(args.data, "clean.h5"), device, args.bs)
    res["oa"]["clean"] = float(oa_clean)
    res["adagam_dalpha"]["clean"] = st
    print(f"clean            OA={oa_clean:.4f}  (N={n})")
    for c in CORRUPTIONS:
        for lvl in range(5):
            key = f"{c}_{lvl}"
            oa, n, st = evaluate_file(net, os.path.join(args.data, key + ".h5"), device, args.bs)
            res["oa"][key], res["n_points"][key], res["adagam_dalpha"][key] = float(oa), n, st
            print(f"{key:17s}OA={oa:.4f}  (N={n})" + (f"  |d_alpha| per stage={np.round(st, 4)}" if st else ""))

    oa_c = {c: float(np.mean([res["oa"][f"{c}_{l}"] for l in range(5)])) for c in CORRUPTIONS}
    ce_c = {c: 1 - oa_c[c] for c in CORRUPTIONS}
    res["oa_per_corruption"] = oa_c
    res["mean_corrupted_oa"] = float(np.mean(list(oa_c.values())))
    res["ce_dgcnn_norm"] = {c: ce_c[c] / (1 - DGCNN_OA[c]) for c in CORRUPTIONS}
    res["mCE"] = float(np.mean(list(res["ce_dgcnn_norm"].values())))
    res["RmCE"] = float(np.mean([(oa_clean - oa_c[c]) / (DGCNN_OA["clean"] - DGCNN_OA[c]) for c in CORRUPTIONS]))
    if args.ref:
        with open(args.ref) as f:
            ref = json.load(f)
        res["ce_vs_ref"] = {c: ce_c[c] / (1 - ref["oa_per_corruption"][c]) for c in CORRUPTIONS}
        res["mCE_vs_ref"] = float(np.mean(list(res["ce_vs_ref"].values())))

    print("\n" + "-" * 60)
    for c in CORRUPTIONS:
        print(f"{c:16s} mean OA={oa_c[c]:.4f}  CE/CE_DGCNN={res['ce_dgcnn_norm'][c]:.3f}")
    print(f"clean OA={oa_clean:.4f} | mean corrupted OA={res['mean_corrupted_oa']:.4f} | "
          f"mCE={res['mCE']:.3f} | RmCE={res['RmCE']:.3f}"
          + (f" | mCE vs ref={res['mCE_vs_ref']:.3f}" if args.ref else "") + f" | {time.time() - t0:.0f}s")

    with open(os.path.join(args.ckpt, "modelnet_c.json"), "w") as f:
        json.dump(res, f, indent=2)
    with open(os.path.join(args.ckpt, "modelnet_c.csv"), "w") as f:
        f.write("split,oa\n" + "".join(f"{k},{v:.4f}\n" for k, v in res["oa"].items()))


if __name__ == "__main__":
    main()
