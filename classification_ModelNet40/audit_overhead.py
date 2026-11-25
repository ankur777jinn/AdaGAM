"""
Parameter / FLOP / latency overhead of AdaGAM vs the static PointMLP baseline.

Usage:
    python audit_overhead.py --model pointMLP
    python audit_overhead.py --model pointMLPElite --reduction 8

FLOPs come from torch.utils.flop_counter (counts matmul/conv FLOPs = 2*MACs; pooling and
elementwise ops are not counted, which is the usual convention). Latency is median wall-clock
of a batch-1 forward pass.
"""

import argparse
import time

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

import models


def count(model, x):
    params = sum(p.numel() for p in model.parameters())
    with FlopCounterMode(display=False) as fc:
        model(x)
    return params, fc.get_total_flops()


@torch.no_grad()
def latency(model, x, reps=50):
    for _ in range(10):
        model(x)
    times = []
    for _ in range(reps):
        if x.is_cuda:
            torch.cuda.synchronize()
        t = time.perf_counter()
        model(x)
        if x.is_cuda:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    return float(np.median(times)) * 1000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="pointMLP")
    p.add_argument("--reduction", default=4, type=int)
    p.add_argument("--use_std", action="store_true")
    p.add_argument("--gam_norm", default="pointmlp")
    p.add_argument("--no_latency", action="store_true")
    a = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    x = torch.rand(1, 3, 1024, device=device)
    base = models.__dict__[a.model]().to(device).eval()
    ada = models.__dict__[a.model](gam="adagam", gam_kwargs=dict(
        reduction=a.reduction, use_std=a.use_std, norm=a.gam_norm)).to(device).eval()

    with torch.no_grad():
        pb, fb = count(base, x)
        pa, fa = count(ada, x)
    print(f"device={device}  model={a.model}  r={a.reduction}  use_std={a.use_std}")
    print(f"Base   : params={pb / 1e6:.4f}M  GFLOPs={fb / 1e9:.3f}")
    print(f"AdaGAM : params={pa / 1e6:.4f}M  GFLOPs={fa / 1e9:.3f}")
    print(f"Overhead: params +{(pa - pb) / pb * 100:.2f}%  FLOPs +{(fa - fb) / fb * 100:.2f}%")
    per_stage = [sum(p.numel() for p in m.parameters()) for m in ada.modules()
                 if m.__class__.__name__ == "AdaGAM"]
    print(f"AdaGAM params per stage: {per_stage}")
    if not a.no_latency:
        lb, la = latency(base, x), latency(ada, x)
        print(f"Latency (bs=1, median): base={lb:.2f}ms  adagam={la:.2f}ms  (+{(la - lb) / lb * 100:.1f}%)")


if __name__ == "__main__":
    main()
