"""Tiny synthetic stand-ins for ModelNet40 and ModelNet-C so every script can be tested on CPU
without downloading anything. The numbers these produce mean NOTHING; they only exercise code paths.

    python tests/fake_data.py --out /tmp/fake      # -> /tmp/fake/modelnet40_ply_hdf5_2048, /tmp/fake/modelnet_c
"""
import argparse
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from corruptions import CORRUPTIONS, apply_corruption  # noqa: E402


def _shape(cls: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """A crude class-dependent shape: an ellipsoid whose axes and a bump depend on the class id."""
    u = rng.normal(size=(n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    axes = np.array([1.0, 0.4 + 0.6 * ((cls * 7) % 10) / 10, 0.3 + 0.7 * ((cls * 3) % 10) / 10])
    pc = u * axes
    pc[:, 2] += 0.2 * np.sin((cls % 5 + 1) * pc[:, 0] * np.pi)
    pc = pc - pc.mean(0)
    return (pc / np.linalg.norm(pc, axis=1).max()).astype(np.float32)


def make_modelnet40(root: str, per_class_train: int = 2, per_class_test: int = 1, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    d = os.path.join(root, "modelnet40_ply_hdf5_2048")
    os.makedirs(d, exist_ok=True)
    for part, per in (("train", per_class_train), ("test", per_class_test)):
        data, label = [], []
        for c in range(40):
            for _ in range(per):
                data.append(_shape(c, 2048, rng))
                label.append([c])
        with h5py.File(os.path.join(d, f"ply_data_{part}0.h5"), "w") as f:
            f["data"] = np.stack(data)
            f["label"] = np.array(label, dtype=np.uint8)
    return d


def make_modelnet_c(root: str, n: int = 8, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    d = os.path.join(root, "modelnet_c")
    os.makedirs(d, exist_ok=True)
    labels = np.arange(n) % 40
    clean = np.stack([_shape(int(c), 1024, rng) for c in labels])

    def write(name, arr):
        with h5py.File(os.path.join(d, name + ".h5"), "w") as f:
            f["data"] = arr.astype(np.float32)
            f["label"] = labels.reshape(-1, 1).astype(np.int64)

    write("clean", clean)
    for c in CORRUPTIONS:
        for lvl in range(5):
            write(f"{c}_{lvl}", np.stack([apply_corruption(pc, c, lvl, rng) for pc in clean]))
    return d


def make_fake_result(run_dir: str, run: str, gam: str, aug: str = "none", seed: int = 0) -> str:
    """A fake modelnet_c.json + args.json (for figure/summary tests)."""
    import json
    from eval_modelnet_c import compute_metrics
    rng = np.random.default_rng(seed)
    os.makedirs(run_dir, exist_ok=True)
    oa = {"clean": [float(rng.uniform(0.9, 0.93))]}
    for i, c in enumerate(CORRUPTIONS):
        start = rng.uniform(0.75, 0.9)
        oa[c] = [float(start - 0.04 * l * rng.uniform(0.5, 1.5)) for l in range(5)]
    dal = None
    if gam == "adagam":
        dal = {"clean": [[0.01, 0.02, 0.03, 0.02]]}
        for c in CORRUPTIONS:
            dal[c] = [[0.01 + 0.005 * l * (s + 1) * rng.uniform(0.5, 1.5) for s in range(4)] for l in range(5)]
    train_c = ["jitter", "dropout_global", "add_global", "scale"] if aug == "corrupt" else []
    res = {"run": os.path.basename(run_dir), "model": "pointMLPElite", "gam": gam, "aug": aug,
           "train_corruptions": train_c, "which": "last", "epoch": 0, "max_samples": 0,
           "leaderboard_comparable": aug == "none", "oa": oa, "adagam_dalpha": dal}
    res.update(compute_metrics(oa))
    with open(os.path.join(run_dir, "modelnet_c.json"), "w") as f:
        json.dump(res, f, indent=2)
    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump({"model": "pointMLPElite", "gam": gam, "aug": aug, "train_corruptions": train_c,
                   "reduction": 8}, f)
    return run_dir


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=8)
    a = p.parse_args()
    print(make_modelnet40(a.out))
    print(make_modelnet_c(a.out, a.n))
