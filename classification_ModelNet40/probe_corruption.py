"""Novelty #1 - implicit corruption-type linear probe.

Question: do the features AdaGAM conditions on (its context vector g, and its hidden layer h)
tell apart the 8 input conditions (clean + 7 ModelNet-C corruptions), although the network was
never told which corruption it saw? We freeze a trained run, pool g (and h) over the anchors of each
stage, and fit a logistic-regression probe.

Design choices (fixing the old version):
  * balanced classes: every class gets exactly the same objects (n = --samples_per_class);
    severities are assigned round-robin per object, so n < 5 still works (fewer severities used);
  * grouped split by OBJECT id: an object is in train for all 8 conditions or in test for all 8,
    so the probe cannot recognise the object instead of the corruption;
  * number of stages read from the model;
  * a static run gets the same context features (same pooling over the same normalised tensor) for
    a fair comparison; use --use_std / --use_min to match an AdaGAM run built with those flags.

    python probe_corruption.py --ckpt checkpoints/pointMLPElite-adagam-42 --data data/modelnet_c
    python probe_corruption.py --ckpt checkpoints/pointMLPElite-base-42   --data data/modelnet_c
"""
import argparse
import json
import os

import h5py
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from corruptions import CORRUPTIONS
from eval_modelnet_c import load_model

CLASSES = ["clean"] + CORRUPTIONS


def assign(n_total, n_per_class, test_frac, seed):
    rng = np.random.default_rng(seed)
    objs = rng.choice(n_total, size=min(n_per_class, n_total), replace=False)
    n_test = max(1, int(round(len(objs) * test_frac)))
    if n_test >= len(objs):
        raise ValueError("need at least 2 objects (one train, one test); raise --samples_per_class")
    test_objs = set(objs[:n_test].tolist())
    plan = []   # (class_idx, split_name, object_id, severity)
    for ci, c in enumerate(CLASSES):
        for j, o in enumerate(objs):
            sev = 0 if c == "clean" else (j + ci) % 5            # round-robin severities
            plan.append((ci, c if c == "clean" else f"{c}_{sev}", int(o), "test" if o in test_objs else "train"))
    return plan


@torch.no_grad()
def extract(net, data_dir, plan, device, batch_size):
    """Returns {feature_name: array [n_samples, dim]} aligned with plan order."""
    by_split = {}
    for k, (_, split, o, _) in enumerate(plan):
        by_split.setdefault(split, []).append((o, k))
    feats = {}
    for split, items in by_split.items():
        items.sort()                                              # h5py fancy indexing needs increasing order
        idx = np.array([o for o, _ in items])
        with h5py.File(os.path.join(data_dir, f"{split}.h5"), "r") as f:
            pcs = f["data"][idx].astype("float32")
        for b in range(0, len(idx), batch_size):
            x = torch.from_numpy(pcs[b:b + batch_size]).to(device).permute(0, 2, 1)
            _, _, aux = net(x, return_aux=True)
            rows = [k for _, k in items[b:b + batch_size]]
            for s, a in enumerate(aux):
                for name in ("context", "hidden"):
                    if name in a:
                        v = a[name].mean(dim=1).cpu().numpy()      # pool over anchors S -> [B, dim]
                        key = f"stage{s + 1}_{name}"
                        if key not in feats:
                            feats[key] = np.zeros((len(plan), v.shape[1]), dtype=np.float32)
                        feats[key][rows] = v
    return feats


def fit_probe(X, y, train_mask, seed):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0, random_state=seed))
    clf.fit(X[train_mask], y[train_mask])
    pred = clf.predict(X[~train_mask])
    return float((pred == y[~train_mask]).mean()), confusion_matrix(y[~train_mask], pred, labels=range(len(CLASSES)))


def main(argv=None):
    p = argparse.ArgumentParser("Corruption-type linear probe (Novelty #1)")
    p.add_argument("--ckpt", required=True, help="run folder")
    p.add_argument("--data", default=os.path.join("data", "modelnet_c"))
    p.add_argument("--which", default="last", choices=["last", "best"])
    p.add_argument("--samples_per_class", type=int, default=300)
    p.add_argument("--test_frac", type=float, default=0.3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--use_std", action="store_true", help="static runs: add std pooling to the parity context")
    p.add_argument("--use_min", action="store_true", help="static runs: add min pooling to the parity context")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=None, help="default: <ckpt>/probe_corruption.json")
    a = p.parse_args(argv)
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    net, args, _ = load_model(a.ckpt, a.which, device)
    if args.get("gam") != "adagam":
        for g in net.local_grouper_list:
            g.use_std, g.use_min = a.use_std or g.use_std, a.use_min or g.use_min
    with h5py.File(os.path.join(a.data, "clean.h5"), "r") as f:
        n_total = f["data"].shape[0]
    plan = assign(n_total, a.samples_per_class, a.test_frac, a.seed)
    y = np.array([ci for ci, _, _, _ in plan])
    train_mask = np.array([s == "train" for _, _, _, s in plan])
    feats = extract(net, a.data, plan, device, a.batch_size)

    results = {"run": os.path.basename(os.path.normpath(a.ckpt)), "gam": args.get("gam"), "classes": CLASSES,
               "chance": 1.0 / len(CLASSES), "n_objects": a.samples_per_class,
               "n_train": int(train_mask.sum()), "n_test": int((~train_mask).sum()), "probes": {}}
    stages = sorted({k.split("_")[0] for k in feats}, key=lambda s: int(s[5:]))
    for key in sorted(feats, key=lambda k: (int(k.split("_")[0][5:]), k)):
        acc, cm = fit_probe(feats[key], y, train_mask, a.seed)
        results["probes"][key] = {"acc": acc, "dim": int(feats[key].shape[1]), "confusion": cm.tolist()}
        print(f"  {key:18s} dim={feats[key].shape[1]:5d}  test acc = {acc:.3f}  (chance {1 / len(CLASSES):.3f})")
    for name in ("context", "hidden"):
        keys = [f"{s}_{name}" for s in stages if f"{s}_{name}" in feats]
        if keys:
            X = np.concatenate([feats[k] for k in keys], 1)
            acc, cm = fit_probe(X, y, train_mask, a.seed)
            results["probes"][f"all_{name}"] = {"acc": acc, "dim": int(X.shape[1]), "confusion": cm.tolist()}
            print(f"  {'all_' + name:18s} dim={X.shape[1]:5d}  test acc = {acc:.3f}")
    out = a.out or os.path.join(a.ckpt, "probe_corruption.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"==> wrote {out}  (train {results['n_train']} / test {results['n_test']} samples, objects split by id)")
    return results


if __name__ == "__main__":
    main()
