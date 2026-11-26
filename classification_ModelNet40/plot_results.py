"""
Result figures from the ModelNet-C evaluations (eval_modelnet_c.py writes <run>/modelnet_c.json).

    python plot_results.py --runs checkpoints/pointMLP-base-42 checkpoints/pointMLP-adagam-42 \
        checkpoints/pointMLP-base_aug-42 checkpoints/pointMLP-adagam_aug-42 \
        --data data/modelnet_c --out results/figures

  fig_oa_vs_severity.png     accuracy vs severity, one panel per corruption, one line per run
  fig_ce_bars.png            corruption error relative to DGCNN (1.0) per corruption, mCE in legend
  fig_dalpha_heatmap.png     AdaGAM's mean |delta-alpha| per stage x corruption x severity (AdaGAM runs)
  fig_corruption_gallery.png one test object under every corruption and severity (needs --data)

A --runs entry may be a run folder or a modelnet_c.json path.
"""

import argparse
import json
import os

import numpy as np

from xai_common import COLORS, CORRUPTIONS, SHORT, INK_2, mpl_style

LABEL = {"pointMLP-base-42": "PointMLP", "pointMLP-adagam-42": "AdaGAM",
         "pointMLP-base_aug-42": "PointMLP (corruption-trained)",
         "pointMLP-adagam_aug-42": "AdaGAM (corruption-trained)"}


def style_of(name):
    base = COLORS["AdaGAM"] if "adagam" in name else COLORS["PointMLP"]
    aug = "_aug" in name
    return dict(color=base, ls="--" if aug else "-", marker="s" if aug else "o", alpha=0.75 if aug else 1.0)


def load(path):
    p = path if path.endswith(".json") else os.path.join(path, "modelnet_c.json")
    with open(p) as f:
        r = json.load(f)
    name = os.path.basename(r.get("run", os.path.dirname(p)).rstrip("/\\"))
    r["name"], r["label"] = name, LABEL.get(name, name)
    return r


def fig_oa(runs, out, plt):
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharey=True)
    for ax, c in zip(axes.flat, CORRUPTIONS + ["clean"]):
        if c == "clean":
            ax.axis("off")
            for r in runs:
                ax.plot([], [], label=f"{r['label']} (clean {r['oa']['clean'] * 100:.1f}%)", **style_of(r["name"]))
            ax.legend(loc="center", fontsize=9)
            continue
        for r in runs:
            ax.plot(range(5), [r["oa"][f"{c}_{s}"] * 100 for s in range(5)], lw=2, ms=4, **style_of(r["name"]))
        ax.set_title(SHORT[c]); ax.set_xticks(range(5)); ax.set_ylim(0, 100)
        ax.set_xlabel("Severity")
    axes[0, 0].set_ylabel("Accuracy (%)"); axes[1, 0].set_ylabel("Accuracy (%)")
    fig.suptitle("ModelNet-C accuracy vs severity (all 2,468 test objects)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_oa_vs_severity.png"), dpi=200)
    plt.close(fig)


def fig_ce(runs, out, plt):
    fig, ax = plt.subplots(figsize=(13, 4.8))
    w = 0.8 / len(runs)
    x = np.arange(len(CORRUPTIONS))
    for i, r in enumerate(runs):
        st = style_of(r["name"])
        ax.bar(x + (i - (len(runs) - 1) / 2) * w, [r["ce_dgcnn_norm"][c] for c in CORRUPTIONS], w,
               color=st["color"], alpha=st["alpha"], hatch="//" if "_aug" in r["name"] else None,
               edgecolor="white", label=f"{r['label']}  mCE {r['mCE']:.3f}")
    ax.axhline(1.0, color=INK_2, lw=1, ls=":")
    ax.text(len(CORRUPTIONS) - 0.45, 1.02, "DGCNN = 1.0", color=INK_2, fontsize=9, ha="right")
    ax.set_xticks(x); ax.set_xticklabels([SHORT[c] for c in CORRUPTIONS])
    ax.set_ylabel("Corruption error / DGCNN's  (lower is better)")
    ax.set_title("Corruption error relative to DGCNN, averaged over severities", loc="left")
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_ce_bars.png"), dpi=200)
    plt.close(fig)


def fig_dalpha(runs, out, plt):
    ada = [r for r in runs if r.get("adagam_dalpha") and r["adagam_dalpha"].get("scale_0")]
    if not ada:
        return
    fig, axes = plt.subplots(len(ada), 4, figsize=(15, 3.2 * len(ada)), squeeze=False)
    for row, r in zip(axes, ada):
        for st, ax in enumerate(row):
            m = np.array([[r["adagam_dalpha"][f"{c}_{s}"][st] for s in range(5)] for c in CORRUPTIONS])
            im = ax.imshow(m, cmap="Oranges", aspect="auto")
            ax.set_xticks(range(5)); ax.set_yticks(range(len(CORRUPTIONS)))
            ax.set_yticklabels([SHORT[c] for c in CORRUPTIONS] if st == 0 else [])
            ax.set_title(f"{r['label']}: stage {st + 1}" if st == 0 else f"stage {st + 1}", fontsize=10)
            for i in range(m.shape[0]):
                for j in range(5):
                    ax.text(j, i, f"{m[i, j]:.3f}", ha="center", va="center", fontsize=6.5,
                            color="white" if m[i, j] > m.min() + 0.6 * (m.max() - m.min()) else "black")
            ax.grid(False)
            fig.colorbar(im, ax=ax, fraction=0.046)
    for ax in axes[-1]:
        ax.set_xlabel("Severity")
    fig.suptitle("How strongly AdaGAM adapts: mean |Δα| per stage, corruption and severity", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_dalpha_heatmap.png"), dpi=200)
    plt.close(fig)


def fig_gallery(data, idx, out, plt):
    import h5py
    names = ["clean"] + [f"{c}_{s}" for c in CORRUPTIONS for s in range(5)]
    fig = plt.figure(figsize=(14, 17))
    for k, (c) in enumerate(CORRUPTIONS):
        for s in range(5):
            with h5py.File(os.path.join(data, f"{c}_{s}.h5"), "r") as f:
                p = f["data"][idx]
                lab = int(f["label"][idx].reshape(-1)[0])
            ax = fig.add_subplot(len(CORRUPTIONS), 5, k * 5 + s + 1, projection="3d")
            ax.scatter(p[:, 0], p[:, 2], p[:, 1], s=0.6, c=p[:, 1], cmap="Blues", vmin=-1.2, vmax=1)
            ax.set_axis_off(); ax.view_init(elev=20, azim=-60)
            lim = 0.85
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
            ax.set_title(f"{SHORT[c]} {s}  ({len(p)} pts)", fontsize=9)
    fig.suptitle(f"ModelNet-C test object #{idx}: 7 corruptions × 5 severities", fontweight="bold", y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_corruption_gallery.png"), dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--data", default=None, help="ModelNet-C folder, for the corruption gallery")
    p.add_argument("--object", default=305, type=int, help="test object for the gallery")
    p.add_argument("--out", default="results/figures")
    a = p.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mpl_style()
    os.makedirs(a.out, exist_ok=True)
    runs = [load(r) for r in a.runs]
    fig_oa(runs, a.out, plt)
    fig_ce(runs, a.out, plt)
    fig_dalpha(runs, a.out, plt)
    if a.data:
        fig_gallery(a.data, a.object, a.out, plt)
    print("wrote", sorted(os.listdir(a.out)))


if __name__ == "__main__":
    main()
