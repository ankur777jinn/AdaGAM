"""ModelNet-C-style corruptions for TRAINING the `_aug` runs (never used for evaluation).

The definitions follow the official ModelNet-C build script
(github.com/jiawei-ren/ModelNet-C, build_modelnetc/corrupt_utils.py). Evaluation ALWAYS uses the
official pre-built ModelNet-C .h5 files, never these functions.

Differences from the official script, on purpose:
  * every function takes an explicit `rng` (numpy Generator) so the demo can seed corruptions;
  * `add_local` allocates only the points it adds (the official version allocates N points and
    breaks when N < total cluster size);
  * clouds are NOT refilled to 1024 points: dropout returns fewer points and add_* returns more,
    exactly like ModelNet-C. Training batches stay rectangular because the collate function
    (data.CorruptCollate) applies one corruption + severity to the whole batch.

Severity is 0..4 (ModelNet-C levels 1..5).
"""
import math
from typing import Callable, Dict, Optional

import numpy as np

CORRUPTIONS = ["scale", "jitter", "rotate", "dropout_global", "dropout_local", "add_global", "add_local"]
# training-side short names used in older docs / args (drop_*) -> ModelNet-C file names (dropout_*)
ALIASES = {"drop_global": "dropout_global", "drop_local": "dropout_local"}
SEEN_DEFAULT = ["jitter", "dropout_global", "add_global", "scale"]
UNSEEN_DEFAULT = ["dropout_local", "add_local", "rotate"]


def canonical(name: str) -> str:
    name = name.strip()
    name = ALIASES.get(name, name)
    if name not in CORRUPTIONS and name != "clean":
        raise ValueError(f"unknown corruption '{name}'. Valid: {CORRUPTIONS} (+ aliases {list(ALIASES)})")
    return name


def _rng(rng: Optional[np.random.Generator]) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng()


def _check(pc: np.ndarray, level: int) -> np.ndarray:
    pc = np.asarray(pc, dtype=np.float32)
    if pc.ndim != 2 or pc.shape[1] != 3:
        raise ValueError(f"expected [N, 3] point cloud, got {pc.shape}")
    if not 0 <= int(level) <= 4:
        raise ValueError(f"severity must be in 0..4, got {level}")
    return pc


def _pc_normalize(pc: np.ndarray) -> np.ndarray:
    pc = pc - pc.mean(axis=0)
    m = np.max(np.sqrt((pc ** 2).sum(axis=1)))
    return pc / max(m, 1e-8)


def _shuffle(pc: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return pc[rng.permutation(pc.shape[0])]


def _cluster_sizes(rng: np.random.Generator, num_clusters: int, total: int):
    assign = rng.integers(0, num_clusters, size=total)
    return [int((assign == i).sum()) for i in range(num_clusters)]


def _points_in_unit_sphere(rng: np.random.Generator, n: int) -> np.ndarray:
    r = np.power(rng.uniform(0.0, 1.0, (n, 1)), 1 / 3)
    theta = np.arccos(rng.uniform(-1.0, 1.0, (n, 1)))
    phi = rng.uniform(0, 2 * np.pi, (n, 1))
    return np.concatenate([r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta)], 1)


def corrupt_scale(pc, level, rng=None):
    pc, rng = _check(pc, level), _rng(rng)
    s = [1.6, 1.7, 1.8, 1.9, 2.0][level]
    xyz = rng.uniform(1.0 / s, s, size=[3])
    return _pc_normalize(pc * xyz).astype(np.float32)


def corrupt_jitter(pc, level, rng=None):
    pc, rng = _check(pc, level), _rng(rng)
    sigma = 0.01 * (level + 1)
    return (pc + sigma * rng.standard_normal(pc.shape)).astype(np.float32)


def corrupt_rotate(pc, level, rng=None):
    pc, rng = _check(pc, level), _rng(rng)
    clip = math.pi / 6 / 5 * (level + 1)
    a = rng.uniform(-clip, clip, size=3)
    rx = np.array([[1, 0, 0], [0, np.cos(a[0]), -np.sin(a[0])], [0, np.sin(a[0]), np.cos(a[0])]])
    ry = np.array([[np.cos(a[1]), 0, np.sin(a[1])], [0, 1, 0], [-np.sin(a[1]), 0, np.cos(a[1])]])
    rz = np.array([[np.cos(a[2]), -np.sin(a[2]), 0], [np.sin(a[2]), np.cos(a[2]), 0], [0, 0, 1]])
    return (pc @ (rz @ ry @ rx)).astype(np.float32)


def corrupt_dropout_global(pc, level, rng=None):
    pc, rng = _check(pc, level), _rng(rng)
    rate = [0.25, 0.375, 0.5, 0.625, 0.75][level]
    keep = int(pc.shape[0] * (1 - rate))
    return _shuffle(pc, rng)[:keep].astype(np.float32)


def corrupt_dropout_local(pc, level, rng=None):
    pc, rng = _check(pc, level), _rng(rng)
    total = 100 * (level + 1)
    total = min(total, max(pc.shape[0] - 32, 0))           # never drop the whole cloud
    n_clusters = int(rng.integers(1, 8))
    for k in _cluster_sizes(rng, n_clusters, total):
        if k == 0:
            continue
        pc = _shuffle(pc, rng)
        d = ((pc - pc[:1]) ** 2).sum(axis=1)
        pc = pc[np.argsort(-d, kind="stable")]               # farthest first; the k nearest to pc[0] are cut
        pc = pc[: pc.shape[0] - k]
    return pc.astype(np.float32)


def corrupt_add_global(pc, level, rng=None):
    pc, rng = _check(pc, level), _rng(rng)
    n = 10 * (level + 1)
    return np.concatenate([pc, _points_in_unit_sphere(rng, n)], 0).astype(np.float32)


def corrupt_add_local(pc, level, rng=None):
    pc, rng = _check(pc, level), _rng(rng)
    total = 100 * (level + 1)
    n_clusters = int(rng.integers(1, 8))
    pc = _shuffle(pc, rng)
    add = np.zeros((total, 3), dtype=np.float64)              # sized by what we add (works for any N)
    done = 0
    for i, k in enumerate(_cluster_sizes(rng, n_clusters, total)):
        sigma = rng.uniform(0.075, 0.125)
        centre = pc[i % pc.shape[0]]
        add[done:done + k] = centre + sigma * rng.standard_normal((k, 3))
        done += k
    norm2 = (add ** 2).sum(axis=1)
    out = norm2 > 1
    add[out] = add[out] / norm2[out, None]                    # same rule as the official script
    return np.concatenate([pc, add], 0).astype(np.float32)


CORRUPT_FNS: Dict[str, Callable] = {
    "scale": corrupt_scale,
    "jitter": corrupt_jitter,
    "rotate": corrupt_rotate,
    "dropout_global": corrupt_dropout_global,
    "dropout_local": corrupt_dropout_local,
    "add_global": corrupt_add_global,
    "add_local": corrupt_add_local,
}


def apply_corruption(pc: np.ndarray, name: str, level: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    name = canonical(name)
    if name == "clean":
        return np.asarray(pc, dtype=np.float32)
    return CORRUPT_FNS[name](pc, int(level), rng)


def output_size(name: str, level: int, n: int = 1024) -> int:
    """Number of points after a corruption (deterministic for every ModelNet-C corruption)."""
    name = canonical(name)
    if name == "dropout_global":
        return int(n * (1 - [0.25, 0.375, 0.5, 0.625, 0.75][level]))
    if name == "dropout_local":
        return n - min(100 * (level + 1), max(n - 32, 0))
    if name == "add_global":
        return n + 10 * (level + 1)
    if name == "add_local":
        return n + 100 * (level + 1)
    return n


def random_corrupt(pc: np.ndarray, corruptions=None, max_severity: int = 4, rng=None):
    """Pick a corruption and severity uniformly and apply it. Returns (cloud, name, level)."""
    rng = _rng(rng)
    corruptions = [canonical(c) for c in (corruptions or SEEN_DEFAULT)]
    name = corruptions[int(rng.integers(len(corruptions)))]
    level = int(rng.integers(0, max_severity + 1))
    return apply_corruption(pc, name, level, rng), name, level
