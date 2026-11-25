"""Evaluate a trained run on ModelNet-C (Ren et al., ICML 2022): OA per corruption x severity,
CE / mCE (normalised to DGCNN), RCE / RmCE, mCE vs our own baseline (--ref), and AdaGAM's mean
|d_alpha| per stage. Resumable: finished splits are cached in <run>/modelnet_c_partial.json.

    python eval_modelnet_c.py --run checkpoints/pointMLPElite-adagam-42 --which last
    python eval_modelnet_c.py --run checkpoints/pointMLPElite-adagam-42 --ref checkpoints/pointMLPElite-base-42

Metric definitions follow the official modelnetc_utils/eval.py: OA averaged over the 5 severities
and rounded to 3 decimals, then
    CE_c  = (1 - OA_c) / (1 - OA^DGCNN_c)                 mCE  = mean_c CE_c
    RCE_c = (OA_clean - OA_c) / (OA^DGCNN_clean - OA^DGCNN_c)   RmCE = mean_c RCE_c
"""
import argparse
import csv
import json
import os
import time
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import models
from corruptions import CORRUPTIONS
from data import load_h5
from utils import device_report, load_checkpoint, strip_module_prefix

# Verified 2026-11-25 against the official ModelNet-C code (github.com/jiawei-ren/ModelNet-C,
# modelnetc_utils/modelnetc_utils/eval.py, dict DGCNN_OA); the README table gives DGCNN
# mCE 1.000 / clean OA 0.926. Only per-corruption means are published there, not per severity.
DGCNN_OA = {
    "clean": 0.926,
    "scale": 0.906,
    "jitter": 0.684,
    "rotate": 0.785,
    "dropout_global": 0.752,
    "dropout_local": 0.793,
    "add_global": 0.705,
    "add_local": 0.725,
}


def load_model(run_dir: str, which: str = "last", device="cpu"):
    """Build the model described by <run>/args.json and load <which>_checkpoint.pth."""
    with open(os.path.join(run_dir, "args.json")) as f:
        args = json.load(f)
    net = models.build_model(args.get("model", "pointMLPElite"), num_classes=40, gam=args.get("gam", "static"),
                             reduction=args.get("reduction", 8), use_std=args.get("use_std", False),
                             use_min=args.get("use_min", False), gam_norm=args.get("gam_norm", "pointmlp"),
                             bound=args.get("bound", False))
    ckpt = load_checkpoint(os.path.join(run_dir, f"{which}_checkpoint.pth"), map_location="cpu")
    net.load_state_dict(strip_module_prefix(ckpt["net"]))
    net.to(device).eval()
    return net, args, ckpt


@torch.no_grad()
def eval_split(net, path: str, device, batch_size: int, max_samples: int = 0) -> Dict:
    data, label = load_h5(path)
    if max_samples:
        data, label = data[:max_samples], label[:max_samples]
    loader = DataLoader(TensorDataset(torch.from_numpy(data), torch.from_numpy(label)), batch_size=batch_size)
    correct, total = 0, 0
    n_stages = len(net.local_grouper_list)
    dsum = np.zeros(n_stages)
    for x, y in loader:
        x = x.to(device).permute(0, 2, 1)
        logits, _, aux = net(x, return_aux=True)
        correct += (logits.argmax(1).cpu() == y).sum().item()
        b = y.size(0)
        total += b
        for s, a in enumerate(aux):
            dsum[s] += a["dalpha_group"].mean().item() * b        # weighted by batch size
    return {"oa": correct / total, "n": total, "num_points": int(data.shape[1]),
            "dalpha": (dsum / total).tolist()}


def compute_metrics(oa: Dict[str, list], ref: Optional[Dict] = None) -> Dict:
    clean = round(oa["clean"][0], 3)
    out = {"oa_clean": clean, "oa_mean": {}, "ce": {}, "rce": {}}
    for c in CORRUPTIONS:
        m = round(float(np.mean(oa[c])), 3)
        out["oa_mean"][c] = m
        out["ce"][c] = round((1 - m) / (1 - DGCNN_OA[c]), 3)
        out["rce"][c] = round((clean - m) / (DGCNN_OA["clean"] - DGCNN_OA[c]), 3)
    out["mCE"] = round(float(np.mean(list(out["ce"].values()))), 3)
    out["RmCE"] = round(float(np.mean(list(out["rce"].values()))), 3)
    out["mOA"] = round(float(np.mean(list(out["oa_mean"].values()))), 3)
    if ref is not None:
        ce_ref = {c: (1 - out["oa_mean"][c]) / max(1 - ref["oa_mean"][c], 1e-6) for c in CORRUPTIONS}
        out["ce_vs_ref"] = {c: round(v, 3) for c, v in ce_ref.items()}
        out["mCE_vs_ref"] = round(float(np.mean(list(ce_ref.values()))), 3)
    return out


def evaluate_run(run_dir, data_dir, which="last", device="cpu", batch_size=32, ref=None,
                 max_samples=0, force=False, verbose=True):
    net, args, ckpt = load_model(run_dir, which, device)
    partial_path = os.path.join(run_dir, f"modelnet_c_partial_{which}.json")
    partial = {}
    if os.path.exists(partial_path) and not force:
        with open(partial_path) as f:
            partial = json.load(f)
        if partial.get("_max_samples", 0) != max_samples or partial.get("_epoch") != ckpt.get("epoch"):
            partial = {}                                  # stale cache (different checkpoint or subset)
    partial["_max_samples"], partial["_epoch"] = max_samples, ckpt.get("epoch")
    splits = ["clean"] + [f"{c}_{l}" for c in CORRUPTIONS for l in range(5)]
    for split in splits:
        if split in partial:
            continue
        path = os.path.join(data_dir, f"{split}.h5")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found. Download ModelNet-C (see README) into {data_dir}")
        t0 = time.time()
        partial[split] = eval_split(net, path, device, batch_size, max_samples)
        with open(partial_path, "w") as f:
            json.dump(partial, f)
        if verbose:
            r = partial[split]
            print(f"  {split:18s} N={r['num_points']:5d} OA={r['oa']:.4f} |da|={np.round(r['dalpha'], 4).tolist()} ({time.time() - t0:.1f}s)")

    oa = {"clean": [partial["clean"]["oa"]]}
    dal = {"clean": [partial["clean"]["dalpha"]]}
    for c in CORRUPTIONS:
        oa[c] = [partial[f"{c}_{l}"]["oa"] for l in range(5)]
        dal[c] = [partial[f"{c}_{l}"]["dalpha"] for l in range(5)]
    ref_metrics = None
    if ref:
        with open(os.path.join(ref, "modelnet_c.json")) as f:
            ref_metrics = json.load(f)
    res = {
        "run": os.path.basename(os.path.normpath(run_dir)), "model": args.get("model"), "gam": args.get("gam"),
        "aug": args.get("aug", "none"), "train_corruptions": args.get("train_corruptions", []),
        "which": which, "epoch": ckpt.get("epoch"), "max_samples": max_samples,
        "leaderboard_comparable": args.get("aug", "none") == "none",
        "oa": oa, "adagam_dalpha": dal if args.get("gam") == "adagam" else None,
        "dgcnn_oa": DGCNN_OA,
    }
    res.update(compute_metrics(oa, ref_metrics))
    if ref:
        res["ref_run"] = os.path.basename(os.path.normpath(ref))
    with open(os.path.join(run_dir, "modelnet_c.json"), "w") as f:
        json.dump(res, f, indent=2)
    with open(os.path.join(run_dir, "modelnet_c.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["corruption", "sev0", "sev1", "sev2", "sev3", "sev4", "mean_OA", "CE", "RCE"])
        w.writerow(["clean", f"{oa['clean'][0]:.4f}", "", "", "", "", f"{res['oa_clean']:.3f}", "", ""])
        for c in CORRUPTIONS:
            w.writerow([c] + [f"{v:.4f}" for v in oa[c]] + [f"{res['oa_mean'][c]:.3f}", res["ce"][c], res["rce"][c]])
        w.writerow(["mCE", res["mCE"], "RmCE", res["RmCE"]])
    if verbose:
        print(f"==> {res['run']}: clean OA {res['oa_clean']:.3f} | mCE {res['mCE']:.3f} | RmCE {res['RmCE']:.3f}"
              + (f" | mCE vs ref {res['mCE_vs_ref']:.3f}" if ref else "")
              + ("" if res["leaderboard_comparable"] else "  [trained with corruptions: NOT leaderboard-comparable]"))
    return res


def main(argv=None):
    p = argparse.ArgumentParser("ModelNet-C evaluation")
    p.add_argument("--run", required=True, help="checkpoint folder, e.g. checkpoints/pointMLPElite-adagam-42")
    p.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "modelnet_c"))
    p.add_argument("--which", default="last", choices=["last", "best"],
                   help="use 'last' (default); 'best' is selected on the test set and flatters results")
    p.add_argument("--ref", default=None, help="baseline run folder (already evaluated) for mCE vs ref")
    p.add_argument("--eval_batch_size", "--batch_size", dest="batch_size", type=int, default=32)
    p.add_argument("--max_samples", type=int, default=0, help="debug: evaluate only the first n clouds per file")
    p.add_argument("--force", action="store_true", help="ignore the resume cache")
    p.add_argument("--device", default=None)
    a = p.parse_args(argv)
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"==> {device_report(device)}")
    return evaluate_run(a.run, a.data, a.which, device, a.batch_size, a.ref, a.max_samples, a.force)


if __name__ == "__main__":
    main()
