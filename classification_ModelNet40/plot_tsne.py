"""
t-SNE Visualization of Global Features (Baseline vs AdaGAM across Corruptions).

Usage:
    python plot_tsne.py --base checkpoints/pointMLP-base-42/features.npz \
        --adagam checkpoints/pointMLP-adagam-42/features.npz --out tsne_comparison.png
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def run_tsne(features, seed):
    tsne = TSNE(n_components=2, perplexity=30, random_state=seed, init="pca", learning_rate="auto")
    return tsne.fit_transform(features)


def plot_panel(ax, emb, corruptions, title):
    unique_c = ["clean"] + sorted(c for c in set(corruptions) if c != "clean")
    cmap = plt.get_cmap("tab10")
    for idx, cname in enumerate(unique_c):
        mask = corruptions == cname
        marker = "*" if cname == "clean" else "o"
        size = 36 if cname == "clean" else 12
        alpha = 0.9 if cname == "clean" else 0.55
        ax.scatter(
            emb[mask, 0], emb[mask, 1],
            s=size, marker=marker, alpha=alpha,
            color="black" if cname == "clean" else cmap(idx % 10),
            label=cname,
        )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=None, help="baseline features.npz")
    p.add_argument("--adagam", required=True, help="AdaGAM features.npz")
    p.add_argument("--out", default="tsne_comparison.png", help="output PNG path")
    p.add_argument("--seed", default=42, type=int)
    args = p.parse_args()

    panels = []
    if args.base and os.path.isfile(args.base):
        d_base = np.load(args.base)
        panels.append(("PointMLP (static)", run_tsne(d_base["features"], args.seed), d_base["corruptions"]))

    d_ada = np.load(args.adagam)
    panels.append(("PointMLP + AdaGAM", run_tsne(d_ada["features"], args.seed), d_ada["corruptions"]))

    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 5.5))
    if len(panels) == 1:
        axes = [axes]

    for ax, (title, emb, corrs) in zip(axes, panels):
        plot_panel(ax, emb, corrs, title)

    axes[-1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
