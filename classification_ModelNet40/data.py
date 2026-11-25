"""ModelNet40 (training) and ModelNet-C (evaluation) data, Windows + Linux safe.

* No shell calls: downloads use urllib / zipfile / shutil, guarded by a lock file so several
  processes can call download() at once without corrupting the data folder.
* ModelNet40 comes from the standard `modelnet40_ply_hdf5_2048.zip` (2,048 points per shape;
  we use the first 1,024 like upstream PointMLP). The original Stanford URL is dead, so a list of
  mirrors is tried. Override with the env var MODELNET40_URL, or download the zip by hand and
  unzip it into classification_ModelNet40/data/.
* ModelNet-C lives on Google Drive; download_modelnet_c() uses `gdown` if installed, otherwise it
  prints manual instructions. Expected layout: data/modelnet_c/{clean,scale_0..4,...}.h5
"""
import glob
import os
import shutil
import sys
import tempfile
import time
import urllib.request
import zipfile
from typing import List, Optional, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from corruptions import SEEN_DEFAULT, apply_corruption, canonical

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MN40_DIRNAME = "modelnet40_ply_hdf5_2048"
MN40_URLS = [
    # Hugging Face mirror (found Nov 2026; the original Stanford host below is offline)
    "https://huggingface.co/datasets/zhangtao-whu/point_cloud_datasets/resolve/main/pcm/modelnet40_ply_hdf5_2048.zip",
    "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip",
]
MODELNET_C_GDRIVE_ID = "1KE6MmXMtfu_mgxg4qLPdEwVD5As8B0rm"   # from the ModelNet-C README

SHAPE_NAMES = [
    "airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl", "car", "chair", "cone",
    "cup", "curtain", "desk", "door", "dresser", "flower_pot", "glass_box", "guitar", "keyboard", "lamp",
    "laptop", "mantel", "monitor", "night_stand", "person", "piano", "plant", "radio", "range_hood", "sink",
    "sofa", "stairs", "stool", "table", "tent", "toilet", "tv_stand", "vase", "wardrobe", "xbox",
]


class _FileLock:
    """Tiny cross-platform lock based on O_EXCL file creation."""

    def __init__(self, path: str, timeout: float = 3600.0):
        self.path, self.timeout = path, timeout

    def __enter__(self):
        start = time.time()
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        while True:
            try:
                self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(self.fd, str(os.getpid()).encode())
                return self
            except FileExistsError:
                if time.time() - start > self.timeout:
                    raise TimeoutError(f"lock {self.path} held too long; delete it if no download is running")
                time.sleep(2.0)

    def __exit__(self, *exc):
        os.close(self.fd)
        try:
            os.remove(self.path)
        except OSError:
            pass


def _download_file(url: str, dst: str) -> None:
    print(f"[data] downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "AdaGAM-3D/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r, open(dst, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                sys.stdout.write(f"\r[data] {done / 2**20:.0f}/{total / 2**20:.0f} MB")
                sys.stdout.flush()
    print()


def modelnet40_ready(data_dir: str = DATA_DIR) -> bool:
    return len(glob.glob(os.path.join(data_dir, MN40_DIRNAME, "ply_data_test*.h5"))) > 0


def download(data_dir: str = DATA_DIR) -> None:
    """Download + unzip ModelNet40 once (safe to call from several processes)."""
    if modelnet40_ready(data_dir):
        return
    os.makedirs(data_dir, exist_ok=True)
    with _FileLock(os.path.join(data_dir, ".download.lock")):
        if modelnet40_ready(data_dir):           # another process finished while we waited
            return
        urls = [os.environ["MODELNET40_URL"]] if os.environ.get("MODELNET40_URL") else MN40_URLS
        errors = []
        for url in urls:
            tmp_dir = tempfile.mkdtemp(dir=data_dir, prefix=".tmp_")
            zpath = os.path.join(tmp_dir, "modelnet40.zip")
            try:
                _download_file(url, zpath)
                with zipfile.ZipFile(zpath) as zf:
                    zf.extractall(tmp_dir)
                src = os.path.join(tmp_dir, MN40_DIRNAME)
                if not os.path.isdir(src):
                    raise RuntimeError(f"archive from {url} has no {MN40_DIRNAME}/ folder")
                shutil.move(src, os.path.join(data_dir, MN40_DIRNAME))
                return
            except Exception as e:  # try the next mirror
                errors.append(f"{url}: {e}")
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError(
            "Could not download ModelNet40. Tried:\n  " + "\n  ".join(errors) +
            f"\nDownload {MN40_DIRNAME}.zip by hand and unzip it so that "
            f"{os.path.join(data_dir, MN40_DIRNAME, 'ply_data_train0.h5')} exists, "
            "or set MODELNET40_URL to a working mirror.")


def download_modelnet_c(data_dir: str = DATA_DIR) -> str:
    """Fetch ModelNet-C with gdown (pip install gdown). Returns the modelnet_c folder."""
    out_dir = os.path.join(data_dir, "modelnet_c")
    if os.path.exists(os.path.join(out_dir, "clean.h5")):
        return out_dir
    os.makedirs(data_dir, exist_ok=True)
    try:
        import gdown  # type: ignore
    except ImportError:
        raise RuntimeError(
            "ModelNet-C is on Google Drive. Either `pip install gdown` and re-run, or download\n"
            f"  https://drive.google.com/file/d/{MODELNET_C_GDRIVE_ID}/view\n"
            f"and unzip modelnet_c.zip into {data_dir} (so that {out_dir}/clean.h5 exists).")
    with _FileLock(os.path.join(data_dir, ".download_c.lock")):
        if os.path.exists(os.path.join(out_dir, "clean.h5")):
            return out_dir
        zpath = os.path.join(data_dir, "modelnet_c.zip")
        gdown.download(id=MODELNET_C_GDRIVE_ID, output=zpath, quiet=False)
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(data_dir)
        os.remove(zpath)
    if not os.path.exists(os.path.join(out_dir, "clean.h5")):
        raise RuntimeError(f"unzipped ModelNet-C but {out_dir}/clean.h5 is missing; check the archive layout")
    return out_dir


def load_data(partition: str, data_dir: str = DATA_DIR):
    files = sorted(glob.glob(os.path.join(data_dir, MN40_DIRNAME, f"ply_data_{partition}*.h5")))
    if not files:
        raise FileNotFoundError(f"no ModelNet40 {partition} files under {os.path.join(data_dir, MN40_DIRNAME)}")
    all_data, all_label = [], []
    for name in files:
        with h5py.File(name, "r") as f:
            all_data.append(f["data"][:].astype("float32"))
            all_label.append(f["label"][:].astype("int64"))
    return np.concatenate(all_data, 0), np.concatenate(all_label, 0).reshape(-1)


def load_h5(path: str):
    with h5py.File(path, "r") as f:
        return f["data"][:].astype("float32"), f["label"][:].astype("int64").reshape(-1)


def translate_params(rng: np.random.Generator):
    return rng.uniform(2.0 / 3.0, 3.0 / 2.0, size=[3]), rng.uniform(-0.2, 0.2, size=[3])


def translate_pointcloud(pc: np.ndarray, rng: Optional[np.random.Generator] = None, params=None) -> np.ndarray:
    """Upstream PointMLP training augmentation: anisotropic scale + shift."""
    rng = rng if rng is not None else np.random.default_rng()
    s, t = params if params is not None else translate_params(rng)
    return (pc * s + t).astype("float32")


class ModelNet40(Dataset):
    def __init__(self, num_points: int = 1024, partition: str = "train", data_dir: str = DATA_DIR):
        self.data, self.label = load_data(partition, data_dir)
        self.num_points, self.partition = num_points, partition

    def __getitem__(self, item):
        pc = self.data[item][: self.num_points]
        label = self.label[item]
        if self.partition == "train":
            rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))
            pc = translate_pointcloud(pc, rng)
            pc = pc[rng.permutation(pc.shape[0])]
        return pc.astype("float32"), label

    def __len__(self):
        return self.data.shape[0]


class ModelNet40Pair(Dataset):
    """Returns the raw (un-augmented) cloud; CorruptCollate builds the (clean, corrupted) views.

    Order of operations per sample (fixes the old translate-before-corrupt bug):
        corrupted = translate(corrupt(raw))   clean = translate(raw)   (same translation for both)
    so `scale`'s re-normalisation and `add_global`'s unit-sphere points act on the canonical cloud.
    """

    def __init__(self, num_points: int = 1024, partition: str = "train", data_dir: str = DATA_DIR):
        self.data, self.label = load_data(partition, data_dir)
        self.num_points = num_points

    def __getitem__(self, item):
        return self.data[item][: self.num_points].astype("float32"), self.label[item]

    def __len__(self):
        return self.data.shape[0]


class CorruptCollate:
    """Batch-level corruption: one (corruption, severity) per batch keeps the batch rectangular
    without duplicating points (dropout batches really have fewer points, as in ModelNet-C)."""

    def __init__(self, corruptions: Sequence[str] = SEEN_DEFAULT, max_severity: int = 4, seed: Optional[int] = None):
        self.corruptions = [canonical(c) for c in corruptions]
        if not 0 <= max_severity <= 4:
            raise ValueError("max_severity must be in 0..4")
        self.max_severity = max_severity
        self.seed = seed

    def __call__(self, batch):
        # derived from numpy's global RNG, which set_seed() / worker_init_fn seed -> reproducible runs
        base = np.random.randint(0, 2**31 - 1)
        rng = np.random.default_rng(base if self.seed is None else self.seed + base)
        name = self.corruptions[int(rng.integers(len(self.corruptions)))]
        level = int(rng.integers(0, self.max_severity + 1))
        clean, corrupt, labels = [], [], []
        for pc, label in batch:
            params = translate_params(rng)
            c = apply_corruption(pc, name, level, rng)
            x = translate_pointcloud(pc, params=params)
            xc = translate_pointcloud(c, params=params)
            clean.append(x[rng.permutation(x.shape[0])])
            corrupt.append(xc[rng.permutation(xc.shape[0])])
            labels.append(label)
        return (torch.from_numpy(np.stack(clean)), torch.from_numpy(np.stack(corrupt)),
                torch.as_tensor(np.array(labels), dtype=torch.long), name, level)


def worker_init_fn(worker_id: int) -> None:
    seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(seed % (2**31 - 1))


def modelnet_c_splits() -> List[str]:
    """The 36 official ModelNet-C split names (file stems), in evaluation order."""
    from corruptions import CORRUPTIONS
    return ["clean"] + [f"{c}_{l}" for c in CORRUPTIONS for l in range(5)]


if __name__ == "__main__":
    download()
    tr = ModelNet40(1024, "train")
    te = ModelNet40(1024, "test")
    print("train", len(tr), "test", len(te), tr[0][0].shape)
