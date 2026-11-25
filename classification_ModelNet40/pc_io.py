"""Pure-NumPy point-cloud upload parser for the demo: .npy .npz .xyz .txt .csv .ply .obj

Every path ends in `prepare_cloud`, which validates (finite, >= MIN_POINTS, size limits), keeps
x, y, z, centres and scales to the unit sphere (like ModelNet40) and resamples to the model's N.
Errors are raised as UploadError with a message that can be shown to the user as-is.
"""
import os
import struct
from typing import Optional

import numpy as np

MAX_BYTES = 50 * 2**20
MAX_POINTS = 2_000_000
MIN_POINTS = 32
EXTS = (".npy", ".npz", ".xyz", ".txt", ".csv", ".ply", ".obj")


class UploadError(ValueError):
    pass


def _as_xyz(arr) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype == object or not np.issubdtype(arr.dtype, np.number):
        raise UploadError("the file does not contain numeric coordinates")
    arr = np.squeeze(arr)
    if arr.ndim == 2 and arr.shape[0] in (3, 6) and arr.shape[1] > 6:
        arr = arr.T                                   # [3, N] -> [N, 3]
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise UploadError(f"expected an array of shape [N, 3] (or [N, >=3]); got {tuple(np.shape(arr))}")
    return arr[:, :3].astype(np.float64)


def _load_text(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip().replace(",", " ").replace(";", " ")
            if not line or line[0] in "#/%":
                continue
            parts = line.split()
            try:
                rows.append([float(p) for p in parts[:3]])
            except ValueError:
                continue                                 # header or comment line
            if len(rows) > MAX_POINTS:
                raise UploadError(f"more than {MAX_POINTS:,} points")
    rows = [r for r in rows if len(r) == 3]
    if not rows:
        raise UploadError("no 'x y z' rows found in the text file")
    return np.array(rows)


_PLY_TYPES = {"char": "b", "int8": "b", "uchar": "B", "uint8": "B", "short": "h", "int16": "h", "ushort": "H",
              "uint16": "H", "int": "i", "int32": "i", "uint": "I", "uint32": "I", "float": "f", "float32": "f",
              "double": "d", "float64": "d"}


def _load_ply(path):
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise UploadError("not a PLY file (missing 'ply' header)")
        fmt, n_vertex, props, in_vertex = None, None, [], False
        while True:
            line = f.readline()
            if not line:
                raise UploadError("PLY header has no 'end_header'")
            tok = line.decode("ascii", "replace").split()
            if not tok:
                continue
            if tok[0] == "format":
                fmt = tok[1]
            elif tok[0] == "element":
                in_vertex = tok[1] == "vertex"
                if in_vertex:
                    n_vertex = int(tok[2])
            elif tok[0] == "property" and in_vertex:
                if tok[1] == "list":
                    raise UploadError("PLY vertex lists are not supported")
                props.append((tok[2], tok[1]))
            elif tok[0] == "end_header":
                break
        if not n_vertex or not props:
            raise UploadError("PLY file has no vertex element")
        if n_vertex > MAX_POINTS:
            raise UploadError(f"more than {MAX_POINTS:,} points")
        names = [p[0] for p in props]
        if not all(a in names for a in ("x", "y", "z")):
            raise UploadError("PLY vertices have no x, y, z properties")
        idx = [names.index(a) for a in ("x", "y", "z")]
        if fmt == "ascii":
            rows = []
            for _ in range(n_vertex):
                parts = f.readline().split()
                rows.append([float(parts[i]) for i in idx])
            return np.array(rows)
        if fmt not in ("binary_little_endian", "binary_big_endian"):
            raise UploadError(f"unsupported PLY format '{fmt}'")
        endian = "<" if fmt == "binary_little_endian" else ">"
        try:
            dtype = np.dtype([(n, endian + _PLY_TYPES[t]) for n, t in props])
        except KeyError as e:
            raise UploadError(f"unsupported PLY property type {e}")
        buf = f.read(dtype.itemsize * n_vertex)
        if len(buf) < dtype.itemsize * n_vertex:
            raise UploadError("PLY file is truncated")
        v = np.frombuffer(buf, dtype=dtype, count=n_vertex)
        return np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)


def _load_obj(path, n_target=1024, seed=0):
    verts, faces = [], []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("v "):
                p = line.split()
                verts.append([float(p[1]), float(p[2]), float(p[3])])
            elif line.startswith("f "):
                idx = [int(t.split("/")[0]) for t in line.split()[1:]]
                for i in range(1, len(idx) - 1):           # fan triangulation
                    faces.append([idx[0], idx[i], idx[i + 1]])
            if len(verts) > MAX_POINTS:
                raise UploadError(f"more than {MAX_POINTS:,} vertices")
    if not verts:
        raise UploadError("OBJ file has no vertices")
    v = np.array(verts)
    if faces and len(v) < 4 * n_target:                 # sparse mesh: sample its surface instead
        fa = np.array(faces)
        fa = np.where(fa < 0, fa + len(v), fa - 1)          # OBJ indices are 1-based, negative = relative
        fa = fa[(fa >= 0).all(1) & (fa < len(v)).all(1)]
        if len(fa):
            a, b, c = v[fa[:, 0]], v[fa[:, 1]], v[fa[:, 2]]
            area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
            if area.sum() > 0:
                rng = np.random.default_rng(seed)
                t = rng.choice(len(fa), size=4 * n_target, p=area / area.sum())
                r1, r2 = np.sqrt(rng.uniform(size=(len(t), 1))), rng.uniform(size=(len(t), 1))
                return (1 - r1) * a[t] + r1 * (1 - r2) * b[t] + r1 * r2 * c[t]
    return v


def prepare_cloud(pts, n_points: int = 1024, seed: int = 0) -> np.ndarray:
    pts = _as_xyz(pts)
    finite = np.isfinite(pts).all(1)
    pts = pts[finite]
    if len(pts) < MIN_POINTS:
        raise UploadError(f"need at least {MIN_POINTS} valid points, found {len(pts)} (NaN/inf rows are dropped)")
    pts = pts - pts.mean(0)
    scale = np.linalg.norm(pts, axis=1).max()
    if not np.isfinite(scale) or scale < 1e-9:
        raise UploadError("all points are identical; nothing to classify")
    pts = pts / scale
    rng = np.random.default_rng(seed)
    if len(pts) >= n_points:
        pts = pts[rng.choice(len(pts), n_points, replace=False)]
    else:                                              # pad by repeating points with tiny jitter
        extra = pts[rng.choice(len(pts), n_points - len(pts), replace=True)]
        pts = np.concatenate([pts, extra + rng.normal(0, 1e-3, extra.shape)])
    return pts.astype(np.float32)


def load_point_cloud(path: Optional[str], n_points: int = 1024, seed: int = 0) -> np.ndarray:
    if not path:
        raise UploadError("no file uploaded")
    ext = os.path.splitext(path)[1].lower()
    if ext not in EXTS:
        raise UploadError(f"unsupported file type '{ext}'. Use one of: {', '.join(EXTS)}")
    if os.path.getsize(path) > MAX_BYTES:
        raise UploadError(f"file is larger than {MAX_BYTES // 2**20} MB")
    try:
        if ext == ".npy":
            arr = np.load(path, allow_pickle=False)
        elif ext == ".npz":
            with np.load(path, allow_pickle=False) as z:
                keys = [k for k in ("points", "pc", "data", "xyz", "arr_0") if k in z.files] or list(z.files)
                if not keys:
                    raise UploadError("the .npz file is empty")
                arr = z[keys[0]]
                if arr.ndim == 3:                       # e.g. a [B, N, 3] batch -> first cloud
                    arr = arr[0]
        elif ext in (".xyz", ".txt", ".csv"):
            arr = _load_text(path)
        elif ext == ".ply":
            arr = _load_ply(path)
        else:
            arr = _load_obj(path, n_points, seed)
    except UploadError:
        raise
    except (ValueError, OSError, struct.error, IndexError, EOFError) as e:
        if ext in (".npy", ".npz"):
            raise UploadError(f"not a valid NumPy {ext} file (object/pickled arrays are refused for safety)")
        raise UploadError(f"could not read the file ({type(e).__name__}: {e})")
    return prepare_cloud(arr, n_points, seed)
