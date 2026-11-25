"""Collect every checkpoints/*/modelnet_c.json into results.md and results.csv.

Seen / unseen columns: for runs trained with --aug corrupt, the seen set is read from the run's
own args.json (train_corruptions); the unseen set is every other ModelNet-C corruption. Clean-only
runs saw no corruption, so for them the columns are computed over the default split
(seen = scale, jitter, dropout_global, add_global; unseen = dropout_local, add_local, rotate) to make
the comparison with the _aug runs like-for-like, and are marked "(ref split)".
"""
import argparse
import csv
import glob
import json
import os

import numpy as np

from corruptions import CORRUPTIONS, SEEN_DEFAULT, canonical

SHORT = {"scale": "Scale", "jitter": "Jitter", "rotate": "Rotate", "dropout_global": "Drop-G",
         "dropout_local": "Drop-L", "add_global": "Add-G", "add_local": "Add-L"}
ORDER = ["base", "adagam", "base_aug", "adagam_aug"]


def _run_key(name):
    for i, k in enumerate(ORDER):
        if f"-{k}-" in name:
            return (i, name)
    return (len(ORDER), name)


def seen_unseen(res, run_dir):
    seen = None
    args_path = os.path.join(run_dir, "args.json")
    aug = res.get("aug", "none")
    if os.path.exists(args_path):
        with open(args_path) as f:
            a = json.load(f)
        aug = a.get("aug", aug)
        if aug == "corrupt":
            seen = [canonical(c) for c in a.get("train_corruptions", SEEN_DEFAULT)]
    ref_split = seen is None
    if seen is None:
        seen = list(SEEN_DEFAULT)
    unseen = [c for c in CORRUPTIONS if c not in seen]
    s = float(np.mean([res["oa_mean"][c] for c in seen])) if seen else float("nan")
    u = float(np.mean([res["oa_mean"][c] for c in unseen])) if unseen else float("nan")
    return s, u, seen, unseen, ref_split


def collect(root):
    rows = []
    for path in sorted(glob.glob(os.path.join(root, "*", "modelnet_c.json")), key=lambda p: _run_key(os.path.basename(os.path.dirname(p)))):
        run_dir = os.path.dirname(path)
        with open(path) as f:
            res = json.load(f)
        s, u, seen, unseen, ref_split = seen_unseen(res, run_dir)
        rows.append({
            "run": res["run"], "model": res.get("model"), "gam": res.get("gam"), "aug": res.get("aug", "none"),
            "which": res.get("which"), "epoch": res.get("epoch"), "comparable": res.get("leaderboard_comparable", True),
            "clean": res["oa_clean"], "mCE": res["mCE"], "RmCE": res["RmCE"], "mOA": res.get("mOA"),
            "mCE_vs_ref": res.get("mCE_vs_ref"),
            **{c: res["oa_mean"][c] for c in CORRUPTIONS},
            "seen_OA": round(s, 3), "unseen_OA": round(u, 3), "seen": seen, "unseen": unseen,
            "ref_split": ref_split, "max_samples": res.get("max_samples", 0),
        })
    return rows


def _best(rows, key, higher):
    vals = [round(r[key], 3) for r in rows if r.get(key) is not None and not np.isnan(r[key])]
    if not vals:
        return None
    return max(vals) if higher else min(vals)


def write(rows, out_md, out_csv):
    cols = ["clean", "mCE", "RmCE"] + CORRUPTIONS + ["seen_OA", "unseen_OA"]
    higher = {c: True for c in cols}
    higher.update({"mCE": False, "RmCE": False})
    best = {c: _best(rows, c, higher[c]) for c in cols}

    def fmt(r, c):
        v = r.get(c)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "–"
        s = f"{v:.3f}"
        return f"**{s}**" if best[c] is not None and round(v, 3) == best[c] and len(rows) > 1 else s

    lines = ["# ModelNet-C results", "",
             "OA = overall accuracy (higher is better). mCE / RmCE are normalised to DGCNN = 1.000 (lower is better).",
             "Per-corruption columns are OA averaged over the 5 severities. Checkpoint: the one named in `which` "
             "(use `last`; `best` is picked on the test set).", "",
             "| Run | Model | GAM | Train | Ckpt | Clean OA | mCE | RmCE | " + " | ".join(SHORT[c] for c in CORRUPTIONS)
             + " | Seen OA | Unseen OA | mCE vs base |",
             "|" + "---|" * (8 + len(CORRUPTIONS) + 3)]
    for r in rows:
        train = "clean" if r["aug"] == "none" else "clean+corrupt ⚠"
        vs_ref = "–" if r["mCE_vs_ref"] is None else f"{r['mCE_vs_ref']:.3f}"
        lines.append(f"| {r['run']} | {r['model']} | {r['gam']} | {train} | {r['which']}@{r['epoch']} | "
                     + " | ".join(fmt(r, c) for c in ["clean", "mCE", "RmCE"] + CORRUPTIONS)
                     + f" | {fmt(r, 'seen_OA')} | {fmt(r, 'unseen_OA')}"
                     + (" (ref split)" if r["ref_split"] else "")
                     + f" | {vs_ref} |")
    lines += ["",
              "**Headline (protocol-compliant):** clean-trained `base` vs clean-trained `adagam`.",
              "",
              "⚠ Runs marked *clean+corrupt* were trained with ModelNet-C-style corruptions, which the ModelNet-C protocol "
              "forbids for leaderboard numbers. They are **not leaderboard-comparable**; for them only the *Unseen OA* "
              "column (corruptions never seen in training) is a generalisation result.",
              "",
              "Seen / unseen sets per run:"]
    for r in rows:
        lines.append(f"- `{r['run']}`: seen = {', '.join(r['seen'])}; unseen = {', '.join(r['unseen'])}"
                     + (" (reference split; this run trained on clean data only)" if r["ref_split"] else ""))
    if any(r["max_samples"] for r in rows):
        lines += ["", "⚠ Some runs were evaluated on a subset (`--max_samples`); not final numbers."]
    lines += ["", "One seed per run: treat OA differences below ~0.5% as within run-to-run noise."]
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    fields = ["run", "model", "gam", "aug", "which", "epoch", "comparable", "clean", "mCE", "RmCE", "mOA",
              "mCE_vs_ref"] + CORRUPTIONS + ["seen_OA", "unseen_OA", "ref_split", "max_samples"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main(argv=None):
    p = argparse.ArgumentParser("Summarise ModelNet-C results")
    p.add_argument("--root", default="checkpoints")
    p.add_argument("--out_md", default="results.md")
    p.add_argument("--out_csv", default="results.csv")
    a = p.parse_args(argv)
    rows = collect(a.root)
    if not rows:
        raise SystemExit(f"no modelnet_c.json under {a.root}/*/ - run eval_modelnet_c.py first")
    write(rows, a.out_md, a.out_csv)
    print(f"wrote {a.out_md} and {a.out_csv} ({len(rows)} runs)")
    return rows


if __name__ == "__main__":
    main()
