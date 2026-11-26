"""
Build the static, in-browser live demo (GitHub Pages) into --out.

    python build_web.py --data <modelnet_c dir> --base checkpoints/pointMLP-base-42 \
        --adagam checkpoints/pointMLP-adagam-42 [--adagam_aug checkpoints/pointMLP-adagam_aug-42] --out web_build

  1. exports the trained models (PointMLP, AdaGAM, optionally the corruption-trained AdaGAM) to ONNX (export_onnx.py) and checks them against PyTorch
  2. picks demo objects: the first test object of each class that both models get right on
     clean data, plus --extra object ids (showcase cases)
  3. packs every official ModelNet-C version of those objects (clean + 7 x 5) as int16
     coordinates, with a mask marking injected points in the add_* files
  4. runs the exported models on everything packed (exactly as the page will) and stores the
     predictions, used to pick the "try these" presets and to self-check the page
  5. copies the page itself from web/
"""

import argparse
import json
import os
import shutil

import h5py
import numpy as np
import onnxruntime as ort

import export_onnx as ex

CORR = ["scale", "jitter", "rotate", "dropout_global", "dropout_local", "add_global", "add_local"]
CONDS = [("clean", 0)] + [(c, s) for c in CORR for s in range(5)]
CLASSES = ["airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl", "car", "chair", "cone",
           "cup", "curtain", "desk", "door", "dresser", "flower_pot", "glass_box", "guitar", "keyboard", "lamp",
           "laptop", "mantel", "monitor", "night_stand", "person", "piano", "plant", "radio", "range_hood", "sink",
           "sofa", "stairs", "stool", "table", "tent", "toilet", "tv_stand", "vase", "wardrobe", "xbox"]


def fname(c, s):
    return "clean.h5" if c == "clean" else f"{c}_{s}.h5"


class Runner:
    def __init__(self, base_path, ada_path, aug_path=None):
        so = ort.SessionOptions()
        self.base = ort.InferenceSession(base_path, so, providers=["CPUExecutionProvider"])
        self.ada = ort.InferenceSession(ada_path, so, providers=["CPUExecutionProvider"])
        self.aug = ort.InferenceSession(aug_path, so, providers=["CPUExecutionProvider"]) if aug_path else None

    @staticmethod
    def _feed(sess, xyz, adapt):
        ins = ex.inputs_for(xyz, adapt)
        have = {i.name for i in sess.get_inputs()}
        return {n: t.numpy() for n, t in zip(ex.NAMES, ins) if n in have}

    def __call__(self, xyz):
        b = self.base.run(None, self._feed(self.base, xyz, 1.0))[0][0]
        a = self.ada.run(None, self._feed(self.ada, xyz, 1.0))[0][0]
        o = self.ada.run(None, self._feed(self.ada, xyz, 0.0))[0][0]
        if self.aug is None:
            return b, a, o
        g = self.aug.run(None, self._feed(self.aug, xyz, 1.0))[0][0]
        return b, a, o, g


def top5(p):
    i = np.argsort(p)[::-1][:5]
    return [[int(k), round(float(p[k]), 4)] for k in i]


def injected_mask(clean, pts):
    """True for points with no clean point within 2.5x the clean cloud's median spacing."""
    d_cc = np.sqrt(((clean[:, None] - clean[None]) ** 2).sum(-1))
    np.fill_diagonal(d_cc, np.inf)
    thr = 2.5 * np.median(d_cc.min(1))
    d = np.sqrt(((pts[:, None] - clean[None]) ** 2).sum(-1)).min(1)
    return d > thr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--adagam", required=True)
    p.add_argument("--adagam_aug", default=None, help="corruption-trained AdaGAM (optional 4th model)")
    p.add_argument("--out", default="web_build")
    p.add_argument("--extra", default="305,320,1089,1156,2333,103,2425,967,768,2216,2314,733,173,1913")
    a = p.parse_args()

    out = a.out
    os.makedirs(os.path.join(out, "models"), exist_ok=True)
    os.makedirs(os.path.join(out, "objects"), exist_ok=True)

    # 1. models
    from eval_modelnet_c import load_model
    import torch
    torch.set_grad_enabled(False)
    rng = np.random.RandomState(1)
    clouds = [rng.randn(n, 3).astype(np.float32) * 0.4 for n in (1024, 1074, 524)]
    paths = {}
    models = [("pointmlp", a.base), ("adagam", a.adagam)] + ([("adagam_aug", a.adagam_aug)] if a.adagam_aug else [])
    for key, ck in models:
        net, _ = load_model(ck, "cpu")
        net.eval()
        paths[key] = os.path.join(out, "models", f"{key}.onnx")
        ex.export(net, paths[key])
        ex.check(net, paths[key], clouds)
    run = Runner(paths["pointmlp"], paths["adagam"], paths.get("adagam_aug"))

    # 2. objects
    files = {c: h5py.File(os.path.join(a.data, fname(*c)), "r") for c in CONDS}
    clean = files[("clean", 0)]["data"][:].astype(np.float32)
    label = files[("clean", 0)]["label"][:].reshape(-1).astype(int)
    chosen = []
    for k in range(40):
        for i in np.where(label == k)[0]:
            b, ad = run(clean[i])[:2]
            if b.argmax() == k and ad.argmax() == k:
                chosen.append(int(i))
                break
    extra = [int(x) for x in a.extra.split(",") if x.strip()]
    ids = sorted(set(chosen) | set(extra))
    print(f"{len(ids)} objects")

    # 3 + 4. pack and predict
    manifest = {"models": [k for k, _ in models], "classes": CLASSES, "conditions": [f"{c}_{s}" if c != "clean" else "clean" for c, s in CONDS],
                "objects": []}
    preds = {}
    for n, i in enumerate(ids):
        parts, meta, scale = [], [], 0.0
        arrays = [files[c]["data"][i].astype(np.float32) for c in CONDS]
        scale = float(max(np.abs(x).max() for x in arrays)) * 1.0001
        blob = bytearray()
        for (c, s), x in zip(CONDS, arrays):
            q = np.round(x / scale * 32767).astype(np.int16)
            xq = q.astype(np.float32) * (scale / 32767)          # exactly what the page decodes
            mask = injected_mask(arrays[0], x) if c.startswith("add") else np.zeros(len(x), bool)
            meta.append({"n": int(len(x)), "inj": int(mask.sum())})
            blob += q.tobytes() + np.packbits(mask).tobytes()
            preds[f"{i}/{c if c == 'clean' else f'{c}_{s}'}"] = [top5(r) for r in run(xq)]
        with open(os.path.join(out, "objects", f"{i}.bin"), "wb") as f:
            f.write(bytes(blob))
        manifest["objects"].append({"id": i, "label": int(label[i]), "scale": scale, "conds": meta})
        print(f"  [{n + 1}/{len(ids)}] object {i} ({CLASSES[label[i]]}) {len(blob) / 1e3:.0f} kB")

    # presets: adaptation decides the answer
    presets = []
    for key, v in preds.items():
        i, cond = key.split("/")
        y = next(o["label"] for o in manifest["objects"] if o["id"] == int(i))
        b, ad, o = (r[0][0] == y for r in v[:3])
        g = len(v) > 3 and v[3][0][0] == y
        if cond.startswith("add_global") and ad and not b and not o:
            presets.append({"id": int(i), "cond": cond, "kind": "rescue", "conf": v[1][0][1]})
        if cond.startswith("jitter") and o and not ad:
            presets.append({"id": int(i), "cond": cond, "kind": "hurt", "conf": v[2][0][1], "cured": bool(g)})
    presets.sort(key=lambda r: (not r.get("cured", False), -r["conf"]))
    manifest["presets"] = presets
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, separators=(",", ":"))
    with open(os.path.join(out, "expected.json"), "w") as f:
        json.dump(preds, f, separators=(",", ":"))
    print(f"presets: {sum(p['kind'] == 'rescue' for p in presets)} rescue, {sum(p['kind'] == 'hurt' for p in presets)} hurt")

    # 5. page
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
    for f in os.listdir(here):
        src = os.path.join(here, f)
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(out, f))
    print("built", out)


if __name__ == "__main__":
    main()
