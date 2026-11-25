"""
Training-time corruptions (numpy, per sample), modelled on the 7 ModelNet-C corruption
families (Ren et al., ICML 2022). Parameter ranges follow ModelNet-C's severity levels
approximately. They are used ONLY for training augmentation; evaluation always uses the
official ModelNet-C .h5 files.

Every function maps [N,3] -> [N,3] (same N) so batches stay rectangular: point-removing
corruptions refill by duplicating surviving points, point-adding ones replace random points.
"""

import numpy as np

ALL_CORRUPTIONS = ["jitter", "drop_global", "drop_local", "add_global", "add_local", "scale", "rotate"]

# severity 0..4 parameters (approximate ModelNet-C levels)
_JITTER_SIGMA = [0.01, 0.02, 0.03, 0.04, 0.05]
_DROP_GLOBAL = [0.25, 0.375, 0.5, 0.625, 0.75]
_DROP_LOCAL = [100, 200, 300, 400, 500]
_ADD_GLOBAL = [10, 20, 30, 40, 50]
_ADD_LOCAL = [100, 200, 300, 400, 500]
_SCALE = [1.6, 1.7, 1.8, 1.9, 2.0]
_ROTATE = [np.pi / 30, np.pi / 15, np.pi / 10, np.pi / 7.5, np.pi / 6]


def _refill(pc, keep_idx, n):
    kept = pc[keep_idx]
    if len(kept) >= n:
        return kept[:n]
    extra = kept[np.random.randint(0, len(kept), n - len(kept))]
    return np.concatenate([kept, extra], axis=0)


def _normalize(pc):
    pc = pc - pc.mean(axis=0, keepdims=True)
    return pc / (np.max(np.linalg.norm(pc, axis=1)) + 1e-8)


def jitter(pc, s):
    return pc + np.random.randn(*pc.shape).astype(pc.dtype) * _JITTER_SIGMA[s]


def drop_global(pc, s):
    n = len(pc)
    keep = np.random.permutation(n)[: int(n * (1 - _DROP_GLOBAL[s]))]
    return _refill(pc, keep, n)


def drop_local(pc, s):
    n = len(pc)
    total = _DROP_LOCAL[s]
    n_clusters = np.random.randint(1, 9)
    sizes = np.random.multinomial(total, np.ones(n_clusters) / n_clusters)
    alive = np.ones(n, dtype=bool)
    for k in sizes:
        idx_alive = np.where(alive)[0]
        if k == 0 or len(idx_alive) <= k:
            continue
        c = pc[np.random.choice(idx_alive)]
        d = np.linalg.norm(pc[idx_alive] - c, axis=1)
        alive[idx_alive[np.argsort(d)[:k]]] = False
    return _refill(pc, np.where(alive)[0], n)


def add_global(pc, s):
    pc = pc.copy()
    m = _ADD_GLOBAL[s]
    pts = np.random.uniform(-1, 1, (m, 3)).astype(pc.dtype)
    pc[np.random.choice(len(pc), m, replace=False)] = pts
    return pc


def add_local(pc, s):
    pc = pc.copy()
    total = _ADD_LOCAL[s]
    n_clusters = np.random.randint(1, 9)
    sizes = np.random.multinomial(total, np.ones(n_clusters) / n_clusters)
    new = []
    for k in sizes:
        if k == 0:
            continue
        c = pc[np.random.randint(len(pc))]
        sigma = np.random.uniform(0.075, 0.125)
        new.append(c + np.random.randn(k, 3) * sigma)
    new = np.concatenate(new, 0).astype(pc.dtype)
    pc[np.random.choice(len(pc), len(new), replace=False)] = new
    return pc


def scale(pc, s):
    f = np.random.uniform(1.0 / _SCALE[s], _SCALE[s], 3).astype(pc.dtype)
    return _normalize(pc * f).astype(pc.dtype)


def rotate(pc, s):
    a = _ROTATE[s]
    ax, ay, az = np.random.uniform(-a, a, 3)
    rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    return (pc @ (rz @ ry @ rx).T).astype(pc.dtype)


_FUNCS = {f.__name__: f for f in [jitter, drop_global, drop_local, add_global, add_local, scale, rotate]}


def random_corrupt(pc, allowed, max_severity=4):
    """Apply one corruption type from `allowed` at a random severity in [0, max_severity]."""
    name = allowed[np.random.randint(len(allowed))]
    s = np.random.randint(0, max_severity + 1)
    out = _FUNCS[name](pc, s)
    np.random.shuffle(out)
    return out.astype("float32")
