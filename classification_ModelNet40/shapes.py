"""Procedural, correctly-labelled test shapes for the demo (works with no dataset on disk).

Each shape is built from primitive surfaces sampled in proportion to their area, then centred and
scaled to the unit sphere like ModelNet40. Class names are ModelNet40 names (so a trained model's
prediction can be compared with the label), but these are toy shapes, not ModelNet40 objects.
"""
from typing import Callable, Dict, List

import numpy as np


def _normalize(pc: np.ndarray) -> np.ndarray:
    pc = pc - pc.mean(0)
    return (pc / max(np.linalg.norm(pc, axis=1).max(), 1e-8)).astype(np.float32)


def _box(rng, c, s, n):
    """Surface of an axis-aligned box: centre c, size s."""
    c, s = np.asarray(c, float), np.asarray(s, float)
    areas = np.array([s[1] * s[2], s[1] * s[2], s[0] * s[2], s[0] * s[2], s[0] * s[1], s[0] * s[1]])
    face = rng.choice(6, size=n, p=areas / areas.sum())
    u = rng.uniform(-0.5, 0.5, size=(n, 3))
    axis = face // 2
    u[np.arange(n), axis] = np.where(face % 2 == 0, -0.5, 0.5)
    return c + u * s


def _cylinder(rng, c, r, h, n, caps=True, axis=1):
    side = 2 * np.pi * r * h
    cap = np.pi * r * r if caps else 0.0
    probs = np.array([side, cap, cap]) / (side + 2 * cap)
    part = rng.choice(3, size=n, p=probs)
    th = rng.uniform(0, 2 * np.pi, n)
    rad = np.where(part == 0, r, r * np.sqrt(rng.uniform(0, 1, n)))
    y = np.where(part == 0, rng.uniform(-h / 2, h / 2, n), np.where(part == 1, -h / 2, h / 2))
    pts = np.stack([rad * np.cos(th), y, rad * np.sin(th)], 1)
    pts = np.roll(pts, axis - 1, axis=1) if axis != 1 else pts
    return np.asarray(c, float) + pts


def _cone(rng, c, r, h, n):
    slant = np.pi * r * np.sqrt(r * r + h * h)
    base = np.pi * r * r
    on_side = rng.uniform(0, 1, n) < slant / (slant + base)
    t = np.sqrt(rng.uniform(0, 1, n))                 # area-uniform along the slant
    th = rng.uniform(0, 2 * np.pi, n)
    rad = np.where(on_side, r * t, r * np.sqrt(rng.uniform(0, 1, n)))
    y = np.where(on_side, h / 2 - h * t, -h / 2)
    return np.asarray(c, float) + np.stack([rad * np.cos(th), y, rad * np.sin(th)], 1)


def _hemisphere(rng, c, r, n, thickness=0.0):
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    v[:, 1] = -np.abs(v[:, 1])                          # lower half -> bowl opening upwards
    rr = r - thickness * rng.integers(0, 2, n)[:, None]
    return np.asarray(c, float) + v * rr


def _split(n, weights):
    w = np.asarray(weights, float)
    k = np.floor(n * w / w.sum()).astype(int)
    k[0] += n - k.sum()
    return k


def airplane(rng, n):
    k = _split(n, [4, 4, 1.2, 0.8])
    return np.concatenate([
        _cylinder(rng, (0, 0, 0), 0.12, 2.0, k[0], axis=0),          # fuselage along x
        _box(rng, (0.05, 0, 0), (0.35, 0.03, 1.8), k[1]),            # wings
        _box(rng, (-0.9, 0.0, 0), (0.2, 0.03, 0.6), k[2]),           # tailplane
        _box(rng, (-0.9, 0.2, 0), (0.25, 0.4, 0.03), k[3]),          # fin
    ])


def bottle(rng, n):
    k = _split(n, [5, 1.5, 1])
    return np.concatenate([_cylinder(rng, (0, -0.3, 0), 0.3, 1.2, k[0]), _cone(rng, (0, 0.45, 0), 0.3, 0.3, k[1]),
                           _cylinder(rng, (0, 0.75, 0), 0.1, 0.35, k[2])])


def bowl(rng, n):
    return _hemisphere(rng, (0, 0, 0), 1.0, n, thickness=0.06)


def chair(rng, n):
    k = _split(n, [3, 3, 1, 1, 1, 1])
    legs = [_box(rng, (x, -0.45, z), (0.07, 0.9, 0.07), kk) for (x, z), kk in
            zip([(-0.4, -0.4), (0.4, -0.4), (-0.4, 0.4), (0.4, 0.4)], k[2:])]
    return np.concatenate([_box(rng, (0, 0, 0), (0.9, 0.08, 0.9), k[0]), _box(rng, (0, 0.5, -0.42), (0.9, 1.0, 0.07), k[1])] + legs)


def cone(rng, n):
    return _cone(rng, (0, 0, 0), 0.6, 1.4, n)


def cup(rng, n):
    k = _split(n, [6, 1])
    body = _cylinder(rng, (0, 0, 0), 0.5, 1.0, k[0], caps=False)
    th = rng.uniform(-np.pi / 2, np.pi / 2, k[1])
    handle = np.stack([0.5 + 0.28 * np.cos(th), 0.28 * np.sin(th), rng.normal(0, 0.03, k[1])], 1)
    return np.concatenate([body, handle])


def stairs(rng, n):
    k = _split(n, [i + 2 for i in range(5)])            # taller steps get more points
    return np.concatenate([_box(rng, (0, -0.5 + 0.1 * (i + 1), -0.4 + 0.2 * i), (1.0, 0.2 * (i + 1), 0.2), kk)
                           for i, kk in enumerate(k)])


def table(rng, n):
    k = _split(n, [4, 1, 1, 1, 1])
    legs = [_box(rng, (x, -0.4, z), (0.08, 0.8, 0.08), kk) for (x, z), kk in
            zip([(-0.8, -0.4), (0.8, -0.4), (-0.8, 0.4), (0.8, 0.4)], k[1:])]
    return np.concatenate([_box(rng, (0, 0.02, 0), (1.9, 0.06, 1.0), k[0])] + legs)


def tent(rng, n):
    u, v = rng.uniform(0, 1, n), rng.uniform(-1, 1, n)
    side = rng.integers(0, 2, n) * 2 - 1
    x = side * (1 - u) * 0.8
    y = u * 1.0 - 0.5
    return np.stack([x, y, v], 1)


def monitor(rng, n):
    k = _split(n, [6, 1, 1.5])
    return np.concatenate([_box(rng, (0, 0.25, 0), (1.6, 1.0, 0.06), k[0]), _box(rng, (0, -0.45, 0), (0.12, 0.4, 0.06), k[1]),
                           _box(rng, (0, -0.66, 0), (0.6, 0.04, 0.35), k[2])])


SHAPES: Dict[str, Callable] = {
    "airplane": airplane, "bottle": bottle, "bowl": bowl, "chair": chair, "cone": cone,
    "cup": cup, "monitor": monitor, "stairs": stairs, "table": table, "tent": tent,
}


def shape_names() -> List[str]:
    return list(SHAPES)


def make_shape(name: str, n: int = 1024, seed: int = 0) -> np.ndarray:
    if name not in SHAPES:
        raise ValueError(f"unknown shape '{name}'. Valid: {shape_names()}")
    rng = np.random.default_rng(seed)
    pc = SHAPES[name](rng, n)
    if pc.shape[0] != n:                       # guard against rounding in _split
        pc = pc[rng.choice(pc.shape[0], n, replace=pc.shape[0] < n)]
    return _normalize(pc[rng.permutation(n)])
