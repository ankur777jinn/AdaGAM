"""Parameter / FLOP / latency audit: static PointMLP vs PointMLP + AdaGAM.

    python audit_overhead.py --model pointMLPElite --reduction 8                 # params + FLOPs + latency
    python audit_overhead.py --model pointMLP --reduction 8 --no_latency          # params + FLOPs only

FLOPs come from torch.utils.flop_counter (counts matmul/conv as 2 x MACs; batch 1, 1024 points).
Latency is batch-1 wall-clock on whatever device you pass and is labelled with that device's name.
Per-stage AdaGAM params count ONLY the AdaGAM modulator (fc1, fc2) - the static affine_alpha/beta are
part of both models and are excluded from the overhead.
"""
import argparse
import json
import time

import torch
from torch.utils.flop_counter import FlopCounterMode

import models


def count(net):
    return sum(p.numel() for p in net.parameters())


def flops(net, n_points=1024, device="cpu"):
    x = torch.randn(1, 3, n_points, device=device)
    net.eval()
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        net(x)
    return fc.get_total_flops()


@torch.no_grad()
def latency(net, device, n_points=1024, warmup=10, iters=50, max_seconds=120.0):
    net.eval().to(device)
    x = torch.randn(1, 3, n_points, device=device)
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)
    t_start = time.time()
    for _ in range(warmup):
        net(x)
        if time.time() - t_start > max_seconds:
            return float("nan"), 0
    sync()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        net(x)
        sync()
        times.append(time.perf_counter() - t0)
        if time.time() - t_start > max_seconds:          # never hang on a slow device
            break
    times.sort()
    return 1000.0 * times[len(times) // 2], len(times)


def main(argv=None):
    p = argparse.ArgumentParser("AdaGAM overhead audit")
    p.add_argument("--model", default="pointMLPElite", choices=sorted(models.MODELS))
    p.add_argument("--reduction", type=int, default=8)
    p.add_argument("--use_std", action="store_true")
    p.add_argument("--use_min", action="store_true")
    p.add_argument("--gam_norm", default="pointmlp", choices=["pointmlp", "patch"])
    p.add_argument("--points", type=int, default=1024)
    p.add_argument("--no_latency", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--max_seconds", type=float, default=120.0, help="time cap per latency measurement")
    p.add_argument("--json", default=None, help="also write the numbers to this file")
    a = p.parse_args(argv)

    kw = dict(reduction=a.reduction, use_std=a.use_std, use_min=a.use_min, gam_norm=a.gam_norm)
    static = models.build_model(a.model, gam="static", gam_norm=a.gam_norm)
    ada = models.build_model(a.model, gam="adagam", **kw)
    ps, pa = count(static), count(ada)
    fs, fa = flops(static, a.points), flops(ada, a.points)
    out = {"model": a.model, "reduction": a.reduction, "params_static": ps, "params_adagam": pa,
           "params_overhead_pct": 100.0 * (pa - ps) / ps, "gflops_static": fs / 1e9, "gflops_adagam": fa / 1e9,
           "flops_overhead_pct": 100.0 * (fa - fs) / fs, "per_stage": []}
    print(f"{a.model} (r={a.reduction}, std={a.use_std}, min={a.use_min}, norm={a.gam_norm}), batch 1, {a.points} points")
    print(f"  params : static {ps / 1e6:.4f}M | AdaGAM {pa / 1e6:.4f}M | +{out['params_overhead_pct']:.2f}%")
    print(f"  GFLOPs : static {fs / 1e9:.3f}  | AdaGAM {fa / 1e9:.3f}  | +{out['flops_overhead_pct']:.2f}%   (FLOPs = 2 x MACs)")
    for i, g in enumerate(ada.local_grouper_list):
        n = count(g.adagam)
        out["per_stage"].append({"stage": i + 1, "channels": g.channels, "hidden": g.adagam.hidden, "params": n})
        print(f"  stage {i + 1}: D={g.channels:4d} hidden={g.adagam.hidden:3d} AdaGAM params={n}")

    if not a.no_latency:
        device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        if device.type == "cuda" and not models.HAS_POINTNET2_OPS:
            print("  note: pointnet2_ops not installed -> pure-PyTorch FPS (latency includes the slower FPS)")
        ls, ns = latency(static, device, a.points, iters=a.iters, max_seconds=a.max_seconds)
        la, na = latency(ada, device, a.points, iters=a.iters, max_seconds=a.max_seconds)
        out.update({"latency_device": name, "latency_ms_static": ls, "latency_ms_adagam": la})
        print(f"  latency (batch 1, median of {min(ns, na)} runs, measured on {name}): "
              f"static {ls:.2f} ms | AdaGAM {la:.2f} ms | +{100 * (la - ls) / ls:.1f}%")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    main()
