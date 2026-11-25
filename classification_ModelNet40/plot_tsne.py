"""t-SNE of global features: static PointMLP vs AdaGAM, clean (faint) vs corrupted (solid).

Reads the files written by dump_features.py:
    python dump_features.py --ckpt checkpoints/pointMLPElite-base-42   --splits clean,add_local_2
    python dump_features.py --ckpt checkpoints/pointMLPElite-adagam-42 --splits clean,add_local_2
    python plot_tsne.py --static checkpoints/pointMLPElite-base-42 --adagam checkpoints/pointMLPElite-adagam-42 \
                        --corruption add_local_2 --out figures/tsne_add_local_2.png

Axes: the two models' features live in different spaces, so no 2-D embedding can make their
coordinates "the same". What we guarantee instead:
  * --mode per_model (default): ONE t-SNE per model fitted on its clean AND corrupted points
    together (so clean vs corrupted are on the same axes within a panel), same seed, same
    perplexity, both feature sets z-scored first;
  * --mode joint: ONE t-SNE over all four sets (both models, z-scored per model).
The panel titles also give numbers that do not depend on the embedding at all:
mean cosine(clean, corrupted) per object and 1-NN accuracy of corrupted features against clean ones.
Colour = class (up to 8, the validated palette; marker shape is a second cue).
"""
import argparse
import os

import numpy as np
from sklearn.manifold import TSNE

import viz_style as vs
from data import SHAPE_NAMES

vs.apply()
import matplotlib.pyplot as plt  # noqa: E402

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def load(run, split):
    path = os.path.join(run, "features", f"features_{split}.npz") if os.path.isdir(run) else run
    if not os.path.exists(path):
        raise SystemExit(f"{path} missing - run dump_features.py --ckpt {run} --splits {split}")
    with np.load(path) as z:
        return z["feats"].astype(np.float32), z["labels"].astype(int)


def zscore(*arrays):
    allx = np.concatenate(arrays)
    mu, sd = allx.mean(0), allx.std(0) + 1e-6
    return [(a - mu) / sd for a in arrays]


def scores(clean, corr, labels):
    cos = np.sum(clean * corr, 1) / (np.linalg.norm(clean, axis=1) * np.linalg.norm(corr, axis=1) + 1e-8)
    d = ((corr[:, None, :] - clean[None, :, :]) ** 2).sum(-1)
    nn = d.argmin(1)
    return float(cos.mean()), float((labels[nn] == labels).mean())


def pick_classes(labels, k):
    vals, counts = np.unique(labels, return_counts=True)
    return vals[np.argsort(-counts, kind="stable")][:k]


def main(argv=None):
    p = argparse.ArgumentParser("t-SNE static vs AdaGAM")
    p.add_argument("--static", required=True, help="static run folder (or a features .npz prefix folder)")
    p.add_argument("--adagam", required=True)
    p.add_argument("--corruption", default="add_local_2", help="ModelNet-C split stem")
    p.add_argument("--classes", type=int, default=8, help="number of classes to show (max 8 colours)")
    p.add_argument("--max_per_class", type=int, default=40)
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--mode", default="per_model", choices=["per_model", "joint"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join("figures", "tsne.png"))
    a = p.parse_args(argv)
    if a.classes > len(vs.SERIES):
        print(f"  note: capping --classes at {len(vs.SERIES)} (beyond 8 hues classes become indistinguishable)")
        a.classes = len(vs.SERIES)

    data = {}
    for key, run in (("static", a.static), ("adagam", a.adagam)):
        fc, lc = load(run, "clean")
        fx, lx = load(run, a.corruption)
        if not np.array_equal(lc, lx):
            raise SystemExit(f"{run}: clean and {a.corruption} labels are not aligned")
        data[key] = (fc, fx, lc)
    labels = data["static"][2]
    if not np.array_equal(labels, data["adagam"][2]):
        raise SystemExit("the two runs' feature files do not cover the same objects")
    classes = pick_classes(labels, a.classes)
    rng = np.random.default_rng(a.seed)
    sel = np.concatenate([rng.permutation(np.where(labels == c)[0])[:a.max_per_class] for c in classes])
    sel.sort()
    lab = labels[sel]
    n = len(sel)
    perp = min(a.perplexity, max(2.0, (2 * n - 1) / 3.0))

    emb, stats = {}, {}
    for key in ("static", "adagam"):
        fc, fx, _ = data[key]
        stats[key] = scores(fc[sel], fx[sel], lab)
    if a.mode == "per_model":
        for key in ("static", "adagam"):
            fc, fx, _ = data[key]
            zc, zx = zscore(fc[sel], fx[sel])
            y = TSNE(2, perplexity=perp, init="pca", random_state=a.seed).fit_transform(np.concatenate([zc, zx]))
            emb[key] = (y[:n], y[n:])
    else:
        blocks = []
        for key in ("static", "adagam"):
            fc, fx, _ = data[key]
            blocks += zscore(fc[sel], fx[sel])
        dims = {b.shape[1] for b in blocks}
        if len(dims) > 1:
            raise SystemExit("--mode joint needs equal feature sizes (same backbone for both runs)")
        y = TSNE(2, perplexity=perp, init="pca", random_state=a.seed).fit_transform(np.concatenate(blocks))
        emb = {"static": (y[:n], y[n:2 * n]), "adagam": (y[2 * n:3 * n], y[3 * n:])}

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))
    for ax, key, title in zip(axes, ("static", "adagam"), ("PointMLP (static GAM)", "PointMLP + AdaGAM")):
        yc, yx = emb[key]
        for i, c in enumerate(classes):
            m = lab == c
            col, mk = vs.SERIES[i], MARKERS[i]
            ax.scatter(yc[m, 0], yc[m, 1], s=22, c=col, marker=mk, alpha=0.22, linewidths=0)
            ax.scatter(yx[m, 0], yx[m, 1], s=26, c=col, marker=mk, edgecolors="white", linewidths=0.6,
                       label=SHAPE_NAMES[c] if c < len(SHAPE_NAMES) else str(c))
        cos, nn = stats[key]
        ax.set_title(f"{title}\nmean cos(clean, corrupted) = {cos:.3f} · 1-NN acc corrupted→clean = {nn:.3f}",
                     loc="left", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9, title="class (solid = corrupted,\nfaint = clean)",
                   title_fontsize=9)
    fig.suptitle(f"t-SNE of global features · {a.corruption} · mode={a.mode} · perplexity {perp:.1f}", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=200)
    plt.close(fig)
    print(f"==> wrote {a.out}; static cos={stats['static'][0]:.3f} nn={stats['static'][1]:.3f} | "
          f"adagam cos={stats['adagam'][0]:.3f} nn={stats['adagam'][1]:.3f}")


if __name__ == "__main__":
    main()
