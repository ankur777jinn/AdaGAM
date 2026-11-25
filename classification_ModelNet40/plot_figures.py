"""The four result figures, from checkpoints/*/modelnet_c.json (+ ModelNet-C .h5 for the gallery).

    python plot_figures.py --root checkpoints --data data/modelnet_c --out figures/
Writes:
    figures/oa_vs_severity.png     7 panels, OA vs severity, one line per run (DGCNN mean as grey reference)
    figures/ce_per_corruption.png  CE per corruption (DGCNN = 1.0), grouped bars, mCE in the legend
    figures/dalpha_heatmap.png     AdaGAM mean |d_alpha|: corruption x severity, one panel per stage
    figures/corruption_gallery.png one object under clean + 7 corruptions x 5 severities
Each figure also gets a .csv with the plotted numbers (table view).
"""
import argparse
import csv
import glob
import json
import os

import numpy as np

import viz_style as vs
from corruptions import CORRUPTIONS
from data import SHAPE_NAMES, load_h5
from eval_modelnet_c import DGCNN_OA

vs.apply()
import matplotlib.pyplot as plt  # noqa: E402

TITLES = {"scale": "Scale", "jitter": "Jitter", "rotate": "Rotate", "dropout_global": "Drop-Global",
          "dropout_local": "Drop-Local", "add_global": "Add-Global", "add_local": "Add-Local"}


def load_runs(root):
    runs = []
    for p in sorted(glob.glob(os.path.join(root, "*", "modelnet_c.json"))):
        with open(p) as f:
            r = json.load(f)
        r["tag"] = vs.run_tag(r["run"])
        runs.append(r)
    order = {k: i for i, k in enumerate(vs.RUN_SLOT)}
    runs.sort(key=lambda r: (order.get(r["tag"], 99), r["run"]))
    return runs


def label(r):
    s = r["tag"]
    return s + (" (corrupt-trained)" if r.get("aug", "none") != "none" else "")


def fig_oa_vs_severity(runs, out):
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.4), sharey=True)
    axes = axes.ravel()
    sev = np.arange(1, 6)
    rows = []
    for i, c in enumerate(CORRUPTIONS):
        ax = axes[i]
        ax.axhline(DGCNN_OA[c], color=vs.NEUTRAL, lw=1, ls=":", zorder=1)
        for k, r in enumerate(runs):
            ax.plot(sev, r["oa"][c], color=vs.run_color(r["run"], k), ls=vs.run_linestyle(r["gam"]),
                    marker="o", ms=5, mec="white", mew=1, label=label(r), zorder=3)
            rows.append([r["run"], c] + [f"{v:.4f}" for v in r["oa"][c]])
        ax.set_title(TITLES[c])
        ax.set_xticks(sev)
        if i % 4 == 0:
            ax.set_ylabel("Overall accuracy")
        if i >= 3:
            ax.set_xlabel("Severity")
    ax = axes[7]
    ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=vs.NEUTRAL, ls=":", lw=1))
    labels.append("DGCNN (mean over severities)")
    ax.legend(handles, labels, loc="center left", fontsize=10, title="Run", title_fontsize=10)
    fig.suptitle("ModelNet-C: accuracy vs corruption severity", x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    _csv(out, ["run", "corruption", "sev1", "sev2", "sev3", "sev4", "sev5"], rows)


def fig_ce(runs, out):
    n = len(runs)
    x = np.arange(len(CORRUPTIONS))
    width = min(0.8 / max(n, 1), 0.22)
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.axhline(1.0, color=vs.NEUTRAL, lw=1, ls=":", zorder=1)
    ax.text(len(CORRUPTIONS) - 0.5, 1.01, "DGCNN = 1.0", color=vs.TEXT_2, ha="right", va="bottom", fontsize=9)
    rows = []
    for k, r in enumerate(runs):
        vals = [r["ce"][c] for c in CORRUPTIONS]
        ax.bar(x + (k - (n - 1) / 2) * width, vals, width=width - 0.02, color=vs.run_color(r["run"], k),
               edgecolor="white", linewidth=1, label=f"{label(r)} · mCE {r['mCE']:.3f}", zorder=2,
               hatch=None if r["gam"] == "adagam" else "//")
        rows.append([r["run"]] + [f"{v:.3f}" for v in vals] + [f"{r['mCE']:.3f}"])
    ax.set_xticks(x, [TITLES[c] for c in CORRUPTIONS])
    ax.set_ylabel("Corruption error (lower is better)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), fontsize=9, ncol=2, borderaxespad=0)
    fig.suptitle("Per-corruption CE, normalised to DGCNN (hatched = static GAM)", x=0.01, ha="left", fontsize=12)
    ax.grid(axis="x", visible=False)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=200)
    plt.close(fig)
    _csv(out, ["run"] + CORRUPTIONS + ["mCE"], rows)


def fig_dalpha(runs, out):
    ada = [r for r in runs if r.get("adagam_dalpha")]
    if not ada:
        print("  (no AdaGAM run with adagam_dalpha - skipping heatmap)")
        return False
    n_stages = len(ada[0]["adagam_dalpha"]["clean"][0])
    fig, axes = plt.subplots(len(ada), n_stages, figsize=(3.2 * n_stages + 1, 3.3 * len(ada)), squeeze=False)
    rows = []
    for i, r in enumerate(ada):
        d = r["adagam_dalpha"]
        mats = np.array([[d[c][l] for l in range(5)] for c in CORRUPTIONS])     # [7, 5, stages]
        clean = np.array(d["clean"][0])
        for s in range(n_stages):
            ax = axes[i, s]
            m = mats[:, :, s]
            vmax = max(m.max(), clean[s]) or 1e-6
            im = ax.imshow(m, cmap=vs.SEQ_CMAP, vmin=0, vmax=vmax, aspect="auto")
            ax.set_xticks(range(5), [str(l + 1) for l in range(5)])
            ax.set_yticks(range(len(CORRUPTIONS)), [TITLES[c] for c in CORRUPTIONS] if s == 0 else [])
            ax.grid(False)
            ax.set_title(f"{r['tag']} · stage {s + 1}\nclean = {clean[s]:.4f}", fontsize=9, loc="left")
            ax.set_xlabel("Severity")
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.outline.set_visible(False)
            cb.ax.tick_params(labelsize=7)
            for ci, c in enumerate(CORRUPTIONS):
                rows.append([r["run"], s + 1, c] + [f"{v:.5f}" for v in m[ci]])
    fig.suptitle("AdaGAM adaptation strength: mean |Δα| per corruption and severity", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    _csv(out, ["run", "stage", "corruption", "sev1", "sev2", "sev3", "sev4", "sev5"], rows)
    return True


def fig_gallery(data_dir, out, index=0):
    clean, labels = load_h5(os.path.join(data_dir, "clean.h5"))
    fig = plt.figure(figsize=(12.5, 19))
    rows = [("clean", None)] + [(c, l) for c in CORRUPTIONS for l in range(5)]

    def draw(ax, pts, title):
        ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1], s=1.6, c=vs.SERIES[0], depthshade=False, linewidths=0)
        ax.set_axis_off()
        ax.set_box_aspect((1, 1, 1))
        lim = 0.72
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.view_init(elev=20, azim=-60)
        ax.set_title(title, fontsize=8, color=vs.TEXT_2, pad=0)

    meta = []
    for r, c in enumerate(CORRUPTIONS):
        for l in range(5):
            pts = load_h5(os.path.join(data_dir, f"{c}_{l}.h5"))[0][index]
            ax = fig.add_subplot(8, 5, (r + 1) * 5 + l + 1, projection="3d")
            draw(ax, pts, f"{TITLES[c]} · sev {l + 1} · N={len(pts)}")
            meta.append([c, l + 1, len(pts)])
    ax = fig.add_subplot(8, 5, 3, projection="3d")
    draw(ax, clean[index], f"Clean · N={len(clean[index])}")
    name = SHAPE_NAMES[int(labels[index])] if int(labels[index]) < len(SHAPE_NAMES) else str(labels[index])
    fig.suptitle(f"ModelNet-C corruption gallery · test object {index} ({name})", x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    _csv(out, ["corruption", "severity", "num_points"], meta)


def _csv(png_path, header, rows):
    with open(os.path.splitext(png_path)[0] + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main(argv=None):
    p = argparse.ArgumentParser("Result figures")
    p.add_argument("--root", default="checkpoints")
    p.add_argument("--data", default=os.path.join("data", "modelnet_c"), help="ModelNet-C folder (gallery)")
    p.add_argument("--out", default="figures")
    p.add_argument("--gallery_index", type=int, default=0)
    p.add_argument("--skip_gallery", action="store_true")
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    runs = load_runs(a.root)
    made = []
    if runs:
        fig_oa_vs_severity(runs, os.path.join(a.out, "oa_vs_severity.png"))
        fig_ce(runs, os.path.join(a.out, "ce_per_corruption.png"))
        made += ["oa_vs_severity.png", "ce_per_corruption.png"]
        if fig_dalpha(runs, os.path.join(a.out, "dalpha_heatmap.png")):
            made.append("dalpha_heatmap.png")
    else:
        print(f"  no modelnet_c.json under {a.root}/*/ - only the gallery can be drawn")
    if not a.skip_gallery and os.path.exists(os.path.join(a.data, "clean.h5")):
        fig_gallery(a.data, os.path.join(a.out, "corruption_gallery.png"), a.gallery_index)
        made.append("corruption_gallery.png")
    print(f"==> wrote {made} to {a.out}/ ({len(runs)} runs)")
    return made


if __name__ == "__main__":
    main()
