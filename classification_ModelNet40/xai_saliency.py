"""
Addition #1 - "Where does each model look when the input is corrupted?"

For PointMLP and AdaGAM on the same ModelNet-C objects, this computes:

  * Point saliency (Zheng et al., ICCV 2019, "PointCloud Saliency Maps", Eq. 4, alpha = 1):
        r_i = ||x_i - median(x)||,  dL/dr_i = g_i . (x_i - median) / r_i,  s_i = -(dL/dr_i) * r_i^2
    One backward pass; approximates "how much the loss rises if point i is dropped".
  * Focus = 1 - H(p) / ln N, with p = positive saliency normalised to sum 1
    (in the spirit of Levi & Gilboa 2023). 1 = all importance on a few points, 0 = spread evenly.
  * Outlier attention (add_global / add_local only): share of positive saliency that lands on
    the injected points (points far from the clean surface), vs the share expected if saliency
    were uniform. Lower = the model is less distracted by fake points.
  * Faithfulness check: remove the top-k salient points vs k random points and track accuracy.
    If removing salient points hurts far more, the saliency map is trustworthy.

Outputs in --out:
    saliency_results.json, saliency_examples.npz
    fig_focus_shift.png, fig_outlier_attention.png, fig_drop_curves.png
    saliency3d_obj<i>.html  (rotatable side-by-side clouds, PointMLP vs AdaGAM)

Usage:
    python xai_saliency.py --base checkpoints/pointMLP-base-42 --adagam checkpoints/pointMLP-adagam-42
    # quick look: --n 100 --severities 2,4
"""

import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from eval_modelnet_c import load_model
from xai_common import (COLORS, CORRUPTIONS, INK_2, SHORT, diverging_cmap, load_split, mpl_style,
                        seeded, sequential_plotly)

MODELS = ["PointMLP", "AdaGAM"]


# ----------------------------------------------------------------------------- computation
def pcsn_saliency(net, pts, labels, device, seed):
    """pts [B,N,3] numpy -> saliency [B,N], preds [B]."""
    x = torch.from_numpy(pts).to(device).requires_grad_(True)
    y = torch.from_numpy(labels).to(device)
    seeded(seed)
    logits = net(x.permute(0, 2, 1))
    loss = F.cross_entropy(logits, y, reduction="sum")
    (g,) = torch.autograd.grad(loss, x)
    with torch.no_grad():
        c = x.median(dim=1, keepdim=True).values
        d = x - c
        r = d.norm(dim=-1).clamp_min(1e-8)
        dl_dr = (g * d).sum(-1) / r
        s = -dl_dr * r ** 2
    return s.detach().cpu().numpy(), logits.argmax(1).detach().cpu().numpy()


def focus_score(s):
    """s [B,N] -> [B] in [0,1]."""
    pos = np.clip(s, 0, None)
    tot = pos.sum(1, keepdims=True)
    p = np.where(tot > 0, pos / np.maximum(tot, 1e-12), 1.0 / s.shape[1])
    h = -(p * np.log(np.clip(p, 1e-12, None))).sum(1)
    return 1.0 - h / math.log(s.shape[1])


def outlier_mask(pts_corr, pts_clean, chunk=32):
    """True for corrupted points that are far from the clean surface. Returns mask [B,N] or None
    if the clouds are not aligned (then the metric is undefined)."""
    d_corr, tau = [], []
    for i in range(0, len(pts_corr), chunk):
        a = torch.from_numpy(pts_corr[i:i + chunk])
        b = torch.from_numpy(pts_clean[i:i + chunk])
        d_corr.append(torch.cdist(a, b).min(-1).values)                # [b, N]
        dd = torch.cdist(b, b)
        dd.diagonal(dim1=1, dim2=2).fill_(float("inf"))
        tau.append(2.5 * dd.min(-1).values.median(-1).values)          # [b] 2.5 x typical spacing
    d_corr, tau = torch.cat(d_corr), torch.cat(tau).unsqueeze(-1)
    if (d_corr.median(-1).values > tau.squeeze(-1)).float().mean() > 0.2:
        return None
    return (d_corr > tau).numpy()


def outlier_attention(s, mask):
    pos = np.clip(s, 0, None)
    tot = np.maximum(pos.sum(1), 1e-12)
    return float(np.mean((pos * mask).sum(1) / tot)), float(mask.mean())


@torch.no_grad()
def accuracy_after_drop(net, pts, labels, order, k, device, seed, bs):
    """order [B,N]: point indices sorted by removal priority. Removes the first k."""
    keep = np.sort(order[:, k:], axis=1)
    kept = np.take_along_axis(pts, keep[..., None], axis=1)
    correct = 0
    for i in range(0, len(kept), bs):
        seeded(seed + i)
        x = torch.from_numpy(kept[i:i + bs]).to(device).permute(0, 2, 1)
        correct += (net(x).argmax(1).cpu().numpy() == labels[i:i + bs]).sum()
    return correct / len(kept)


def saliency_batched(net, pts, labels, device, seed, bs):
    out_s, out_p = [], []
    for i in range(0, len(pts), bs):
        s, p = pcsn_saliency(net, pts[i:i + bs], labels[i:i + bs], device, seed + i)
        out_s.append(s)
        out_p.append(p)
    return np.concatenate(out_s), np.concatenate(out_p)


# ----------------------------------------------------------------------------- figures
def fig_focus_shift(res, severities, out):
    import matplotlib.pyplot as plt
    mpl_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    mats = []
    for m in MODELS:
        base = res["clean"][m]["focus"]
        mats.append(np.array([[res[f"{c}_{l}"][m]["focus"] - base for l in severities] for c in CORRUPTIONS]))
    vmax = max(np.abs(mats[0]).max(), np.abs(mats[1]).max(), 1e-6)
    for ax, m, mat in zip(axes, MODELS, mats):
        im = ax.imshow(mat, cmap=diverging_cmap(), vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(severities)), [str(l + 1) for l in severities])
        ax.set_yticks(range(len(CORRUPTIONS)), [SHORT[c] for c in CORRUPTIONS])
        ax.set_xlabel("Severity")
        ax.set_title(f"{m}: change in focus vs clean")
        ax.grid(False)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:+.3f}", ha="center", va="center", fontsize=8, color="#0b0b0b")
    cb = fig.colorbar(im, ax=axes, shrink=0.85)
    cb.set_label("focus(corrupted) - focus(clean)")
    fig.suptitle("How corruption shifts where each model looks (0 = same focus as on clean data)",
                 fontsize=12, fontweight="bold")
    fig.savefig(os.path.join(out, "fig_focus_shift.png"), dpi=200)
    plt.close(fig)


def fig_outlier_attention(res, severities, out):
    import matplotlib.pyplot as plt
    mpl_style()
    adds = [c for c in ["add_global", "add_local"] if res.get(f"{c}_{severities[0]}", {}).get("PointMLP", {}).get("outlier_attention") is not None]
    if not adds:
        return
    fig, axes = plt.subplots(1, len(adds), figsize=(5.5 * len(adds), 4), constrained_layout=True, squeeze=False)
    w = 0.38
    for ax, c in zip(axes[0], adds):
        x = np.arange(len(severities))
        for j, m in enumerate(MODELS):
            vals = [100 * res[f"{c}_{l}"][m]["outlier_attention"] for l in severities]
            bars = ax.bar(x + (j - 0.5) * w, vals, w * 0.92, color=COLORS[m], label=m)
            ax.bar_label(bars, fmt="%.0f%%", fontsize=8, color=INK_2, padding=2)
        uni = [100 * res[f"{c}_{l}"]["PointMLP"]["outlier_fraction"] for l in severities]
        ax.plot(x, uni, ls="--", color=INK_2, lw=1.5, marker="o", ms=4, label="If attention were uniform")
        ax.set_xticks(x, [str(l + 1) for l in severities])
        ax.set_xlabel("Severity")
        ax.set_ylabel("Share of saliency on injected points (%)")
        ax.set_title(f"{SHORT[c]}: attention stolen by fake points")
        ax.legend(loc="upper left", fontsize=8)
    fig.savefig(os.path.join(out, "fig_outlier_attention.png"), dpi=200)
    plt.close(fig)


def fig_drop_curves(curves, out):
    import matplotlib.pyplot as plt
    mpl_style()
    panels = list(curves.keys())
    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4), constrained_layout=True, squeeze=False)
    for ax, key in zip(axes[0], panels):
        cv = curves[key]
        for m in MODELS:
            ax.plot(cv["ks"], [100 * a for a in cv[m]["salient"]], color=COLORS[m], marker="o", ms=5,
                    label=f"{m}: remove most salient")
            ax.plot(cv["ks"], [100 * a for a in cv[m]["random"]], color=COLORS[m], ls="--", lw=1.5,
                    label=f"{m}: remove random")
        ax.set_xlabel("Points removed (of 1024)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"Faithfulness check on {key}")
        ax.legend(fontsize=8)
    fig.suptitle("Removing the points the saliency map marks as important should hurt much more than random removal",
                 fontsize=11, fontweight="bold")
    fig.savefig(os.path.join(out, "fig_drop_curves.png"), dpi=200)
    plt.close(fig)


def fig_3d(examples, names, out):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    rows = [k for k in examples if k != "labels"]
    n_obj = examples["labels"].shape[0]
    for o in range(n_obj):
        titles = []
        for r in rows:
            for m in MODELS:
                pred, lab = int(examples[r][m]["preds"][o]), int(examples["labels"][o])
                ok = "correct" if pred == lab else "wrong"
                titles.append(f"{m} | {r} | pred {names[pred]} ({ok})")
        fig = make_subplots(rows=len(rows), cols=2, specs=[[{"type": "scene"}] * 2] * len(rows),
                            subplot_titles=titles, vertical_spacing=0.03, horizontal_spacing=0.02)
        for i, r in enumerate(rows):
            for j, m in enumerate(MODELS):
                p = examples[r]["points"][o]
                s = np.clip(examples[r][m]["saliency"][o], 0, None)
                s = s / max(np.quantile(s, 0.99), 1e-12)
                fig.add_trace(go.Scatter3d(
                    x=p[:, 0], y=p[:, 2], z=p[:, 1], mode="markers",
                    marker=dict(size=2.6, color=np.clip(s, 0, 1), colorscale=sequential_plotly(), cmin=0, cmax=1,
                                showscale=(i == 0 and j == 1),
                                colorbar=dict(title="saliency", len=0.3, y=0.9)),
                    hovertemplate="saliency %{marker.color:.2f}<extra></extra>", showlegend=False),
                    row=i + 1, col=j + 1)
        scene = dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                     aspectmode="data", bgcolor="#fcfcfb")
        fig.update_scenes(**scene)
        fig.update_layout(height=330 * len(rows), width=1000, paper_bgcolor="#fcfcfb",
                          title=f"Where each model looks - object {o} (true class: {names[int(examples['labels'][o])]})",
                          font=dict(size=11), margin=dict(l=10, r=10, t=70, b=10))
        fig.write_html(os.path.join(out, f"saliency3d_obj{o}.html"), include_plotlyjs="cdn")


# ----------------------------------------------------------------------------- main
MODELNET40_NAMES = [
    "airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl", "car", "chair", "cone", "cup",
    "curtain", "desk", "door", "dresser", "flower_pot", "glass_box", "guitar", "keyboard", "lamp", "laptop",
    "mantel", "monitor", "night_stand", "person", "piano", "plant", "radio", "range_hood", "sink", "sofa",
    "stairs", "stool", "table", "tent", "toilet", "tv_stand", "vase", "wardrobe", "xbox"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="PointMLP checkpoint dir")
    p.add_argument("--adagam", required=True, help="AdaGAM checkpoint dir")
    p.add_argument("--data", default="data/modelnet_c")
    p.add_argument("--out", default="xai_out/saliency")
    p.add_argument("--which", default="last", choices=["last", "best"])
    p.add_argument("--n", type=int, default=400, help="test objects per split (same objects everywhere)")
    p.add_argument("--severities", default="0,1,2,3,4", help="ModelNet-C levels 0..4 (shown as 1..5)")
    p.add_argument("--drop_ks", default="0,50,100,200,300,400,500")
    p.add_argument("--drop_on", default="clean,add_local_2", help="splits for the faithfulness curves")
    p.add_argument("--examples", default="clean,add_local_4,dropout_local_4,jitter_4")
    p.add_argument("--n_examples", type=int, default=4)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    os.makedirs(a.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nets = {"PointMLP": load_model(a.base, device, a.which)[0], "AdaGAM": load_model(a.adagam, device, a.which)[0]}
    sev = [int(s) for s in a.severities.split(",")]

    clean_all, lab_all = load_split(a.data, "clean", 0)
    rng = np.random.RandomState(a.seed)
    idx = np.sort(rng.choice(len(clean_all), size=min(a.n, len(clean_all)), replace=False))
    clean, labels = clean_all[idx], lab_all[idx]

    res, cache = {}, {}
    splits = [("clean", 0)] + [(c, l) for c in CORRUPTIONS for l in sev]
    for c, l in splits:
        key = "clean" if c == "clean" else f"{c}_{l}"
        pts, lab = (clean, labels) if c == "clean" else load_split(a.data, c, l, idx)
        assert (lab == labels).all(), f"{key}: labels differ from clean.h5 - object order mismatch"
        mask = outlier_mask(pts, clean) if c.startswith("add") else None
        res[key] = {}
        for m, net in nets.items():
            s, preds = saliency_batched(net, pts, lab, device, a.seed, a.bs)
            entry = {"acc": float((preds == lab).mean()), "focus": float(focus_score(s).mean())}
            if mask is not None:
                entry["outlier_attention"], entry["outlier_fraction"] = outlier_attention(s, mask)
            res[key][m] = entry
            cache[(key, m)] = (pts, s, preds)
        line = " | ".join(f"{m}: acc {res[key][m]['acc']:.3f} focus {res[key][m]['focus']:.3f}"
                          + (f" outlier-att {res[key][m]['outlier_attention']:.3f}" if "outlier_attention" in res[key][m] else "")
                          for m in MODELS)
        print(f"{key:18s} {line}")

    # faithfulness curves
    ks = [int(k) for k in a.drop_ks.split(",")]
    curves = {}
    for key in [k.strip() for k in a.drop_on.split(",") if k.strip()]:
        if (key, "PointMLP") not in cache:
            continue
        curves[key] = {"ks": ks}
        for m, net in nets.items():
            pts, s, _ = cache[(key, m)]
            ks_ok = [k for k in ks if pts.shape[1] - k >= 200]
            by_sal = np.argsort(-s, axis=1)
            by_rand = np.argsort(np.random.RandomState(1).rand(*s.shape), axis=1)
            curves[key]["ks"] = ks_ok
            curves[key][m] = {
                "salient": [float(accuracy_after_drop(net, pts, labels, by_sal, k, device, a.seed, a.bs)) for k in ks_ok],
                "random": [float(accuracy_after_drop(net, pts, labels, by_rand, k, device, a.seed, a.bs)) for k in ks_ok]}
        print(f"drop curve {key}: " + " | ".join(
            f"{m} salient {curves[key][m]['salient'][-1]:.3f} vs random {curves[key][m]['random'][-1]:.3f} at k={curves[key]['ks'][-1]}"
            for m in MODELS))

    # examples for the 3D viewer
    ex = {"labels": labels[:a.n_examples]}
    for key in [k.strip() for k in a.examples.split(",") if k.strip()]:
        if (key, "PointMLP") not in cache:
            continue
        ex[key] = {"points": cache[(key, "PointMLP")][0][:a.n_examples]}
        for m in MODELS:
            ex[key][m] = {"saliency": cache[(key, m)][1][:a.n_examples], "preds": cache[(key, m)][2][:a.n_examples]}

    with open(os.path.join(a.out, "saliency_results.json"), "w") as f:
        json.dump({"n": int(len(idx)), "severities_shown": [l + 1 for l in sev], "grid": res, "drop_curves": curves}, f, indent=2)
    np.savez_compressed(os.path.join(a.out, "saliency_examples.npz"), examples=np.array(ex, dtype=object))

    fig_focus_shift(res, sev, a.out)
    fig_outlier_attention(res, sev, a.out)
    if curves:
        fig_drop_curves(curves, a.out)
    fig_3d(ex, MODELNET40_NAMES, a.out)
    print(f"\nWrote figures and results to {a.out}")


if __name__ == "__main__":
    main()
