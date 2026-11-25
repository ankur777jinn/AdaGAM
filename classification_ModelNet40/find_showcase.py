"""
Find ModelNet-C test objects worth showing in the live demo (demo_app.py).

For each corruption/severity it lists objects where
  rescue : PointMLP wrong, AdaGAM right, AdaGAM with adaptation off wrong
           (the adaptive part is what gets this object right)
  hurts  : AdaGAM wrong, AdaGAM with adaptation off right
           (the adaptation is what gets it wrong, e.g. jitter)

Predictions use exactly the demo's inference (one object at a time, seeded point sampling),
so every listed object behaves the same way when you type its number into the demo.

    python find_showcase.py --base checkpoints/pointMLP-base-42 \
        --adagam checkpoints/pointMLP-adagam-42 --data data/modelnet_c
    -> showcase.md
"""

import argparse

import numpy as np
import torch

from demo_app import FPS_SEED, MODELNET40_CLASSES, OfficialData, _infer_probs
from eval_modelnet_c import load_model
from xai_common import set_adaptation

DEFAULT_SETS = "add_global:2,add_global:3,add_global:4,jitter:2,jitter:3,jitter:4"


@torch.no_grad()
def batch_preds(net, pts, device, bs):
    """Fast first pass to shortlist objects; the shortlist is re-checked one by one."""
    out = []
    for i in range(0, len(pts), bs):
        x = torch.from_numpy(pts[i:i + bs]).to(device).permute(0, 2, 1)
        torch.manual_seed(FPS_SEED)
        out.append(net(x).argmax(1).cpu().numpy())
    return np.concatenate(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--adagam", required=True)
    p.add_argument("--data", default="data/modelnet_c")
    p.add_argument("--sets", default=DEFAULT_SETS, help="corruption:severity list (demo names or file names)")
    p.add_argument("--top", default=8, type=int, help="objects listed per category")
    p.add_argument("--bs", default=32, type=int)
    p.add_argument("--out", default="showcase.md")
    a = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base, _ = load_model(a.base, device)
    ada, _ = load_model(a.adagam, device)
    data = OfficialData(a.data)
    alias = {"dropout_global": "drop_global", "dropout_local": "drop_local"}

    lines = ["# Demo showcase objects", "",
             "Type the object number into the demo, pick the same corruption and severity, "
             "keep *Corrupted input* on **Official ModelNet-C files**.", ""]
    for item in a.sets.split(","):
        corr, sev = item.strip().split(":")
        corr = alias.get(corr, corr)
        got = data.get(corr, int(sev))
        if got is None:
            print(f"skip {corr}:{sev} (file not found)")
            continue
        pts, lab = got

        pb = batch_preds(base, pts, device, a.bs)
        set_adaptation(ada, True)
        pa = batch_preds(ada, pts, device, a.bs)
        set_adaptation(ada, False)
        po = batch_preds(ada, pts, device, a.bs)
        set_adaptation(ada, True)
        cand = np.where((pb != lab) | (pa != lab) | (po != lab))[0]

        rescue, hurts = [], []
        for i in cand:
            b, _ = _infer_probs(base, pts[i], device)
            on, _ = _infer_probs(ada, pts[i], device, disable_adapt=False)
            off, _ = _infer_probs(ada, pts[i], device, disable_adapt=True)
            y = int(lab[i])
            ok_b, ok_on, ok_off = b.argmax() == y, on.argmax() == y, off.argmax() == y
            row = (int(i), MODELNET40_CLASSES[y],
                   f"{MODELNET40_CLASSES[int(b.argmax())]} {b.max() * 100:.0f}%",
                   f"{MODELNET40_CLASSES[int(on.argmax())]} {on.max() * 100:.0f}%",
                   f"{MODELNET40_CLASSES[int(off.argmax())]} {off.max() * 100:.0f}%")
            if ok_on and not ok_b and not ok_off:
                rescue.append((float(on[y]), row))
            if ok_off and not ok_on:
                hurts.append((float(off[y]), row))

        n = len(lab)
        acc = lambda q: (q == lab).mean() * 100
        print(f"{corr}:{sev}  acc PointMLP {acc(pb):.1f}  AdaGAM {acc(pa):.1f}  off {acc(po):.1f}  "
              f"| rescue {len(rescue)}  hurts {len(hurts)}  (of {n})")
        lines += [f"## {corr}, severity {sev}",
                  f"Whole file: PointMLP {acc(pb):.1f}%, AdaGAM {acc(pa):.1f}%, adaptation off {acc(po):.1f}% "
                  f"({n} objects). Rescued by adaptation: {len(rescue)}. Hurt by adaptation: {len(hurts)}.", ""]
        for title, rows in [("Adaptation rescues it (PointMLP ❌, AdaGAM ✅, off ❌)", rescue),
                            ("Adaptation hurts it (AdaGAM ❌, off ✅)", hurts)]:
            rows = [r for _, r in sorted(rows, key=lambda t: -t[0])[:a.top]]
            lines += [f"**{title}**", ""]
            if not rows:
                lines += ["none", ""]
                continue
            lines += ["| Object # | True class | PointMLP | AdaGAM | AdaGAM, adaptation off |",
                      "|---|---|---|---|---|"]
            lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows] + [""]

    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
