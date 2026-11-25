"""
CPU check for the two explainability analyses, no real data or training needed:
    python smoke_test_xai.py
Builds two tiny pointMLPElite checkpoints (static + AdaGAM with non-zero modulation) and a fake
ModelNet-C folder (consistent object order, add_* files = clean + injected points, dropout_* files
with fewer points), then runs xai_saliency.py and cka_drift.py end to end.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

import h5py
import numpy as np
import torch

import models

N_OBJ = 24
rng = np.random.RandomState(0)


def fake_object(n=1024):
    k = rng.randint(3)
    u = rng.randn(n, 3)
    if k == 0:
        u /= np.linalg.norm(u, axis=1, keepdims=True)               # sphere
    elif k == 1:
        u = rng.uniform(-1, 1, (n, 3)) * [1, 0.3, 0.6]             # box
    else:
        t = rng.uniform(0, 2 * np.pi, n)
        u = np.stack([np.cos(t), rng.uniform(-1, 1, n), np.sin(t)], 1)  # cylinder
    u -= u.mean(0)
    return (u / np.linalg.norm(u, axis=1).max()).astype("float32"), k


def write(path, data, label):
    with h5py.File(path, "w") as f:
        f["data"] = data
        f["label"] = label[:, None]


def main():
    tmp = tempfile.mkdtemp()
    try:
        dd = os.path.join(tmp, "modelnet_c")
        os.makedirs(dd)
        objs = [fake_object() for _ in range(N_OBJ)]
        clean = np.stack([o[0] for o in objs])
        label = np.array([o[1] for o in objs], dtype="int64")
        write(os.path.join(dd, "clean.h5"), clean, label)
        for l in range(5):
            write(os.path.join(dd, f"jitter_{l}.h5"), clean + rng.randn(*clean.shape).astype("float32") * 0.01 * (l + 1), label)
            write(os.path.join(dd, f"scale_{l}.h5"), clean * np.float32(0.9), label)
            write(os.path.join(dd, f"rotate_{l}.h5"), clean[:, :, [2, 1, 0]], label)
            keep = 1024 - 128 * (l + 1)
            write(os.path.join(dd, f"dropout_global_{l}.h5"), clean[:, :keep], label)
            write(os.path.join(dd, f"dropout_local_{l}.h5"), clean[:, :keep], label)
            m = 10 * (l + 1)
            out_g = rng.uniform(-1, 1, (N_OBJ, m, 3)).astype("float32")
            write(os.path.join(dd, f"add_global_{l}.h5"), np.concatenate([clean, out_g], 1), label)
            centre = clean[:, :1, :]
            out_l = (centre + rng.randn(N_OBJ, 100 * (l + 1), 3) * 0.1 + 0.3).astype("float32")
            write(os.path.join(dd, f"add_local_{l}.h5"), np.concatenate([clean, out_l], 1), label)

        for name, gam in [("base", "static"), ("ada", "adagam")]:
            ck = os.path.join(tmp, name)
            os.makedirs(ck)
            torch.manual_seed(1)
            net = models.pointMLPElite(gam=gam, gam_kwargs=dict(reduction=8, use_std=False, norm="pointmlp"))
            for mod in net.modules():
                if mod.__class__.__name__ == "AdaGAM":
                    torch.nn.init.normal_(mod.fc2.weight, std=0.05)   # make adaptation non-trivial
            json.dump(dict(model="pointMLPElite", gam=gam, reduction=8, use_std=False, gam_norm="pointmlp"),
                      open(os.path.join(ck, "args.json"), "w"))
            torch.save({"net": {"module." + k: v for k, v in net.state_dict().items()}},
                       os.path.join(ck, "last_checkpoint.pth"))

        out = os.path.join(tmp, "xai")
        common = ["--base", os.path.join(tmp, "base"), "--adagam", os.path.join(tmp, "ada"), "--data", dd, "--bs", "8"]
        runs = [
            [sys.executable, "xai_saliency.py", *common, "--n", "16", "--severities", "0,4",
             "--drop_ks", "0,100,300", "--drop_on", "clean,add_local_0", "--examples", "clean,add_local_4,dropout_local_4",
             "--n_examples", "2", "--out", os.path.join(out, "saliency")],
            [sys.executable, "cka_drift.py", *common, "--n", "16", "--severities", "0,4", "--out", os.path.join(out, "cka")],
        ]
        for cmd in runs:
            r = subprocess.run(cmd, capture_output=True, text=True)
            print(r.stdout[-1800:])
            if r.returncode != 0:
                print(r.stderr[-3000:])
                sys.exit(f"[FAIL] {cmd[1]}")
            print(f"[PASS] {cmd[1]}\n")
        produced = sorted(os.listdir(os.path.join(out, "saliency")) + os.listdir(os.path.join(out, "cka")))
        print("Produced:", produced)
        keep_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xai_smoke_output")
        shutil.rmtree(keep_dir, ignore_errors=True)
        shutil.copytree(out, keep_dir)
        print(f"Copied example outputs (fake data) to {keep_dir}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
