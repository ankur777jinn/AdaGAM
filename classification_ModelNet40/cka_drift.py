"""
Addition #2 - "Does AdaGAM repair the damage corruption does to its features?"

For every ModelNet-C corruption x severity and every network stage s = 1..4, this measures
how similar the features of a corrupted object are to the features of its clean copy, using
linear CKA (Kornblith et al., ICML 2019):

    CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F),   X, Y column-centred [n, C_s]

X = stage-s features of the clean objects as the model normally computes them (the
representation its classifier was trained on); Y = the same objects, corrupted. 1 = corruption
leaves the representation unchanged (up to rotation/scale of feature space); lower = more drift.
Stage features are the output of each stage's PosExtraction, max-pooled over anchor points
(the same pooling PointMLP's classifier uses), so splits with different point counts compare.

Four variants:
  PointMLP                  baseline
  AdaGAM                    adaptive modulation on
  AdaGAM (adaptation off)   same weights, d_alpha = d_beta = 0 on the corrupted input only
                            -> if drift returns, the modulation is what repaired it
  PointMLP + BN-adapt       BatchNorm statistics re-estimated on each corrupted split
                            (Schneider et al., NeurIPS 2020). Needs a whole batch of the same
                            corruption; AdaGAM adapts one object at a time.

Accuracy of each variant on the same objects is reported too (and an mCE on this subset).

Outputs in --out: cka_results.json, fig_cka_stages.png, fig_cka_gain_heatmap.png, fig_accuracy.png

Usage:
    python cka_drift.py --base checkpoints/pointMLP-base-42 --adagam checkpoints/pointMLP-adagam-42
"""

import argparse
import copy
import json
import os

import numpy as np
import torch

from eval_modelnet_c import DGCNN_OA, load_model
from xai_common import (COLORS, CORRUPTIONS, INK_2, SHORT, diverging_cmap, load_split, mpl_style,
                        seeded, set_adaptation)

VARIANTS = ["PointMLP", "AdaGAM", "AdaGAM (adaptation off)", "PointMLP + BN-adapt"]


def linear_cka(x, y):
    x = x.astype(np.float64) - x.mean(0, keepdims=True)
    y = y.astype(np.float64) - y.mean(0, keepdims=True)
    num = np.linalg.norm(y.T @ x, "fro") ** 2
    den = np.linalg.norm(x.T @ x, "fro") * np.linalg.norm(y.T @ y, "fro")
    return float(num / max(den, 1e-12))


@torch.no_grad()
def stage_features(net, pts, device, seed, bs):
    """Returns (list of 4 arrays [n, C_s], preds [n])."""
    feats, hooks = [], []
    store = {}

    def make_hook(i):
        def hook(_m, _inp, out):
            store[i] = out.amax(dim=-1).cpu()
        return hook

    for i, blk in enumerate(net.pos_blocks_list):
        hooks.append(blk.register_forward_hook(make_hook(i)))
    per_stage = [[] for _ in net.pos_blocks_list]
    preds = []
    try:
        for b in range(0, len(pts), bs):
            seeded(seed + b)
            x = torch.from_numpy(pts[b:b + bs]).to(device).permute(0, 2, 1)
            preds.append(net(x).argmax(1).cpu().numpy())
            for i in range(len(per_stage)):
                per_stage[i].append(store[i].numpy())
    finally:
        for h in hooks:
            h.remove()
    feats = [np.concatenate(s) for s in per_stage]
    return feats, np.concatenate(preds)


@torch.no_grad()
def bn_adapted(base_net, pts, device, seed, bs):
    """Copy of the baseline with BatchNorm running stats re-estimated on `pts` (one pass)."""
    net = copy.deepcopy(base_net)
    net.eval()
    for m in net.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.reset_running_stats()
            m.momentum = None            # cumulative average over the pass
            m.train()
    for b in range(0, len(pts), bs):
        chunk = pts[b:b + bs]
        if len(chunk) < 2:
            continue
        seeded(seed + b)
        net(torch.from_numpy(chunk).to(device).permute(0, 2, 1))
    net.eval()
    return net


# ----------------------------------------------------------------------------- figures
def fig_stages(res, sev, out):
    import matplotlib.pyplot as plt
    mpl_style()
    panels = CORRUPTIONS + ["mean"]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharey=True, constrained_layout=True)
    n_st = len(res["clean_ref_dims"])
    xs = np.arange(1, n_st + 1)
    for ax, c in zip(axes.flat, panels):
        for v in VARIANTS:
            if c == "mean":
                y = np.mean([[res["grid"][f"{cc}_{l}"][v]["cka"] for l in sev] for cc in CORRUPTIONS], axis=(0, 1))
            else:
                y = np.mean([res["grid"][f"{c}_{l}"][v]["cka"] for l in sev], axis=0)
            ax.plot(xs, y, color=COLORS[v], marker="o", ms=6, label=v,
                    ls="--" if "off" in v else "-")
        ax.set_title("All corruptions (mean)" if c == "mean" else SHORT[c])
        ax.set_xticks(xs, [f"Stage {i}" for i in xs])
        ax.set_ylim(0, 1.02)
    for ax in axes[:, 0]:
        ax.set_ylabel("CKA(clean, corrupted)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("How far corruption pushes the features at each stage (1.0 = no drift), averaged over severities",
                 fontsize=12, fontweight="bold")
    fig.savefig(os.path.join(out, "fig_cka_stages.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_gain_heatmap(res, sev, out):
    import matplotlib.pyplot as plt
    mpl_style()
    n_st = len(res["clean_ref_dims"])
    fig, axes = plt.subplots(1, n_st, figsize=(3.3 * n_st + 1, 3.8), constrained_layout=True)
    mats = [np.array([[res["grid"][f"{c}_{l}"]["AdaGAM"]["cka"][s] - res["grid"][f"{c}_{l}"]["PointMLP"]["cka"][s]
                       for l in sev] for c in CORRUPTIONS]) for s in range(n_st)]
    vmax = max(max(np.abs(m).max() for m in mats), 1e-6)
    for s, (ax, mat) in enumerate(zip(np.atleast_1d(axes), mats)):
        im = ax.imshow(mat, cmap=diverging_cmap(), vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(sev)), [str(l + 1) for l in sev])
        ax.set_yticks(range(len(CORRUPTIONS)), [SHORT[c] for c in CORRUPTIONS] if s == 0 else [])
        ax.set_xlabel("Severity")
        ax.set_title(f"Stage {s + 1}")
        ax.grid(False)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center", fontsize=7, color="#0b0b0b")
    cb = fig.colorbar(im, ax=axes, shrink=0.9)
    cb.set_label("CKA gain: AdaGAM - PointMLP")
    fig.suptitle("Where AdaGAM keeps features closer to clean (blue) or further (red)", fontsize=12, fontweight="bold")
    fig.savefig(os.path.join(out, "fig_cka_gain_heatmap.png"), dpi=200)
    plt.close(fig)


def fig_accuracy(res, out):
    import matplotlib.pyplot as plt
    mpl_style()
    groups = CORRUPTIONS + ["mean"]
    fig, ax = plt.subplots(figsize=(13, 4.2), constrained_layout=True)
    w = 0.8 / len(VARIANTS)
    x = np.arange(len(groups))
    for j, v in enumerate(VARIANTS):
        vals = [100 * (res["acc_mean"][v] if g == "mean" else res["acc_per_corruption"][v][g]) for g in groups]
        ax.bar(x + (j - (len(VARIANTS) - 1) / 2) * w, vals, w * 0.92, color=COLORS[v],
               label=f"{v} (mCE {res['mCE_subset'][v]:.3f})")
    ax.set_xticks(x, [SHORT.get(g, "Mean") for g in groups])
    ax.set_ylabel("Accuracy on corrupted objects (%)")
    ax.set_title(f"Accuracy on the same {res['n']} objects, averaged over severities (mCE: DGCNN = 1, lower is better)")
    ax.legend(ncol=2, fontsize=9, loc="lower left")
    ax.set_ylim(0, 100)
    fig.savefig(os.path.join(out, "fig_accuracy.png"), dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--adagam", required=True)
    p.add_argument("--data", default="data/modelnet_c")
    p.add_argument("--out", default="xai_out/cka")
    p.add_argument("--which", default="last", choices=["last", "best"])
    p.add_argument("--n", type=int, default=1000, help="objects per split; CKA wants n >= ~1000")
    p.add_argument("--severities", default="0,1,2,3,4")
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    os.makedirs(a.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = load_model(a.base, device, a.which)[0]
    ada = load_model(a.adagam, device, a.which)[0]
    sev = [int(s) for s in a.severities.split(",")]

    clean_all, lab_all = load_split(a.data, "clean", 0)
    idx = np.sort(np.random.RandomState(a.seed).choice(len(clean_all), size=min(a.n, len(clean_all)), replace=False))
    clean, labels = clean_all[idx], lab_all[idx]

    set_adaptation(ada, True)
    ref = {"PointMLP": stage_features(base, clean, device, a.seed, a.bs)[0],
           "AdaGAM": stage_features(ada, clean, device, a.seed, a.bs)[0]}
    ref["AdaGAM (adaptation off)"] = ref["AdaGAM"]          # reference = the model's normal clean features
    ref["PointMLP + BN-adapt"] = ref["PointMLP"]

    grid = {}
    for c in CORRUPTIONS:
        for l in sev:
            key = f"{c}_{l}"
            pts, lab = load_split(a.data, c, l, idx)
            assert (lab == labels).all(), f"{key}: object order differs from clean.h5"
            grid[key] = {}
            nets = {"PointMLP": base, "AdaGAM": ada, "AdaGAM (adaptation off)": ada,
                    "PointMLP + BN-adapt": bn_adapted(base, pts, device, a.seed, a.bs)}
            for v, net in nets.items():
                set_adaptation(ada, v != "AdaGAM (adaptation off)")
                feats, preds = stage_features(net, pts, device, a.seed, a.bs)
                grid[key][v] = {"cka": [linear_cka(r, f) for r, f in zip(ref[v], feats)],
                                "acc": float((preds == lab).mean())}
            set_adaptation(ada, True)
            print(f"{key:18s} " + " | ".join(
                f"{v.split(' (')[0] if 'off' not in v else 'AdaGAM-off'}: CKA4 {grid[key][v]['cka'][-1]:.3f} acc {grid[key][v]['acc']:.3f}"
                for v in VARIANTS))

    acc_c = {v: {c: float(np.mean([grid[f"{c}_{l}"][v]["acc"] for l in sev])) for c in CORRUPTIONS} for v in VARIANTS}
    res = {
        "n": int(len(idx)), "severities_shown": [l + 1 for l in sev], "pool": "max over anchors",
        "clean_ref_dims": [int(f.shape[1]) for f in ref["PointMLP"]], "grid": grid,
        "acc_per_corruption": acc_c,
        "acc_mean": {v: float(np.mean(list(acc_c[v].values()))) for v in VARIANTS},
        "mCE_subset": {v: float(np.mean([(1 - acc_c[v][c]) / (1 - DGCNN_OA[c]) for c in CORRUPTIONS])) for v in VARIANTS},
        "note": "Accuracies/mCE are on the --n subset and the --severities used; use eval_modelnet_c.py for official numbers.",
    }
    with open(os.path.join(a.out, "cka_results.json"), "w") as f:
        json.dump(res, f, indent=2)
    fig_stages(res, sev, a.out)
    fig_gain_heatmap(res, sev, a.out)
    fig_accuracy(res, a.out)
    print("\nmCE on this subset: " + " | ".join(f"{v}: {res['mCE_subset'][v]:.3f}" for v in VARIANTS))
    print(f"Wrote figures and results to {a.out}")


if __name__ == "__main__":
    main()
