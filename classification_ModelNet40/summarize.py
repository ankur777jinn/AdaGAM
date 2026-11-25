"""
Collect every checkpoints/*/modelnet_c.json into one table for the slides.

    python summarize.py                 -> prints a markdown table, writes results.md and results.csv
"""

import glob
import json
import os

CORR = ["scale", "jitter", "rotate", "dropout_global", "dropout_local", "add_global", "add_local"]
SHORT = ["Scale", "Jitter", "Rotate", "Drop-G", "Drop-L", "Add-G", "Add-L"]
SEEN_DEFAULT = ["jitter", "dropout_global", "add_global", "scale"]
UNSEEN_DEFAULT = ["dropout_local", "add_local", "rotate"]

# True if higher is better, False if lower is better, None if not ranked
HIGHER_BETTER = [None, True, True, False, False, False, True, True] + [True] * len(CORR)

rows = []
for path in sorted(glob.glob("checkpoints/*/modelnet_c.json")):
    run = os.path.basename(os.path.dirname(path))
    r = json.load(open(path))
    oa_c = r["oa_per_corruption"]
    seen_oa = sum(oa_c[c] for c in SEEN_DEFAULT) / len(SEEN_DEFAULT)
    unseen_oa = sum(oa_c[c] for c in UNSEEN_DEFAULT) / len(UNSEEN_DEFAULT)
    rows.append(
        [run, r["oa"]["clean"], r["mean_corrupted_oa"], r["mCE"], r["RmCE"],
         r.get("mCE_vs_ref"), seen_oa, unseen_oa]
        + [oa_c[c] for c in CORR]
    )

hdr = ["Run", "Clean OA", "Mean corr. OA", "mCE (DGCNN=1)", "RmCE",
       "mCE vs baseline", "Seen OA", "Unseen OA"] + SHORT

best_vals = []
for col_idx, hb in enumerate(HIGHER_BETTER):
    vals = [row[col_idx] for row in rows if row[col_idx] is not None]
    if hb is None or len(vals) <= 1:
        best_vals.append(None)
    else:
        best_vals.append(max(vals) if hb else min(vals))

fmt = lambda v: "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def fmt_md(v, col_idx):
    s = fmt(v)
    b = best_vals[col_idx]
    if b is not None and v == b:
        return f"**{s}**"
    return s


md = "| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n"
md += "".join("| " + " | ".join(fmt_md(v, i) for i, v in enumerate(row)) + " |\n" for row in rows)
print(md)
open("results.md", "w").write(md)
open("results.csv", "w").write(",".join(hdr) + "\n" + "".join(",".join(fmt(v) for v in r) + "\n" for r in rows))
