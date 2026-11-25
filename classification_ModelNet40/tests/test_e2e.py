"""End-to-end (opaque-box) checks on CPU with synthetic data. Nothing is downloaded.

    python -m pytest classification_ModelNet40/tests -q      # if pytest is installed
    python classification_ModelNet40/tests/test_e2e.py         # plain Python, same tests
"""
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))      # classification_ModelNet40/
ROOT = os.path.dirname(HERE)                                            # repo root
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402

_TMP = None


def tmp():
    """One shared synthetic dataset for all tests (built on first use)."""
    global _TMP
    if _TMP is None:
        from tests.fake_data import make_modelnet40, make_modelnet_c
        _TMP = tempfile.mkdtemp(prefix="adagam_e2e_")
        make_modelnet40(_TMP)
        make_modelnet_c(_TMP, n=12)
    return _TMP


def py(*args, check=True):
    r = subprocess.run([sys.executable, *args], cwd=HERE, capture_output=True, text=True,
                       env={**os.environ, "PYTHONUNBUFFERED": "1"})
    if check and r.returncode != 0:
        raise AssertionError(f"{args} failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    return r.stdout + r.stderr


# ----------------------------------------------------------------------------- core
def test_smoke_suite():
    out = py("smoke_test.py")
    assert "All smoke tests passed." in out


def test_train_resume_eval_summarize():
    d = tmp()
    ck = os.path.join(d, "ck")
    common = ["--data_dir", d, "--checkpoint_root", ck, "--batch_size", "8", "--max_batches", "2",
              "--num_workers", "0", "--eval_batch_size", "16"]
    py("main.py", "--msg", "base", "--gam", "static", "--epoch", "1", *common)
    py("main.py", "--msg", "adagam", "--gam", "adagam", "--epoch", "1", *common)
    out = py("main.py", "--msg", "adagam", "--gam", "adagam", "--epoch", "2", "--resume", *common)
    assert "resumed from epoch 0" in out and "Epoch 2/2" in out
    py("main.py", "--msg", "adagam_aug", "--gam", "adagam", "--aug", "corrupt", "--epoch", "1", *common)
    run = os.path.join(ck, "pointMLPElite-adagam-42")
    with open(os.path.join(run, "args.json")) as f:
        a = json.load(f)
    assert a["model"] == "pointMLPElite" and a["reduction"] == 8
    mc = os.path.join(d, "modelnet_c")
    py("eval_modelnet_c.py", "--run", os.path.join(ck, "pointMLPElite-base-42"), "--data", mc, "--max_samples", "4")
    py("eval_modelnet_c.py", "--run", run, "--data", mc, "--max_samples", "4",
       "--ref", os.path.join(ck, "pointMLPElite-base-42"))
    py("eval_modelnet_c.py", "--run", os.path.join(ck, "pointMLPElite-adagam_aug-42"), "--data", mc, "--max_samples", "4")
    res = json.load(open(os.path.join(run, "modelnet_c.json")))
    assert res["which"] == "last" and "mCE_vs_ref" in res and len(res["adagam_dalpha"]["scale"]) == 5
    out_md = os.path.join(d, "results.md")
    py("summarize.py", "--root", ck, "--out_md", out_md, "--out_csv", os.path.join(d, "results.csv"))
    md = open(out_md, encoding="utf-8").read()
    assert "Seen OA" in md and "Unseen OA" in md and "not leaderboard-comparable" in md


def test_runner_order_and_shell():
    out = py("run_experiments.py", "train", "--dry_run", "--epochs", "1")
    order = re.findall(r"--msg (\w+)", out)
    assert order == ["base", "adagam", "base_aug", "adagam_aug"], order
    assert "--resume" in out and "--amp" in out
    if shutil.which("bash"):
        r = subprocess.run(["bash", "-n", os.path.join(HERE, "run_experiments.sh")], capture_output=True)
        assert r.returncode == 0, r.stderr


def test_help_lists_flags():
    out = py("main.py", "--help")
    for flag in ["--gam", "--reduction", "--use_std", "--gam_norm", "--aug", "--train_corruptions", "--max_severity",
                 "--consistency", "--cons_type", "--batch_size", "--amp", "--num_workers", "--epoch", "--resume"]:
        assert flag in out, flag


# ----------------------------------------------------------------------- diagnostics
def test_audit_numbers():
    import audit_overhead
    o = audit_overhead.main(["--model", "pointMLP", "--reduction", "8", "--no_latency"])
    assert round(o["params_static"] / 1e6, 4) == 13.2385 and round(o["params_adagam"] / 1e6, 4) == 13.4167
    assert round(o["gflops_static"], 3) == 31.212 and round(o["gflops_adagam"], 3) == 31.245
    e = audit_overhead.main(["--model", "pointMLPElite", "--reduction", "8", "--no_latency"])
    assert round(e["params_static"] / 1e6, 4) == 0.7162 and round(e["params_adagam"] / 1e6, 4) == 0.7644


def _fake_adagam_run(d):
    import models
    run = os.path.join(d, "diag", "pointMLPElite-adagam-42")
    os.makedirs(run, exist_ok=True)
    torch.manual_seed(0)
    net = models.build_model("pointMLPElite", gam="adagam")
    for m in net.adagam_modules():
        torch.nn.init.normal_(m.fc2.weight, std=0.05)
    torch.save({"net": net.state_dict(), "epoch": 0}, os.path.join(run, "last_checkpoint.pth"))
    json.dump({"model": "pointMLPElite", "gam": "adagam", "reduction": 8, "aug": "none"},
              open(os.path.join(run, "args.json"), "w"))
    base = os.path.join(d, "diag", "pointMLPElite-base-42")
    os.makedirs(base, exist_ok=True)
    torch.save({"net": models.build_model("pointMLPElite").state_dict(), "epoch": 0}, os.path.join(base, "last_checkpoint.pth"))
    json.dump({"model": "pointMLPElite", "gam": "static", "aug": "none"}, open(os.path.join(base, "args.json"), "w"))
    return run, base


def test_probe_saliency_tsne_figures():
    d = tmp()
    mc = os.path.join(d, "modelnet_c")
    run, base = _fake_adagam_run(d)
    out = py("probe_corruption.py", "--ckpt", run, "--data", mc, "--samples_per_class", "4", "--test_frac", "0.25")
    assert "chance" in out and os.path.exists(os.path.join(run, "probe_corruption.json"))
    py("probe_corruption.py", "--ckpt", base, "--data", mc, "--samples_per_class", "3")     # n < 5 severities
    sal = os.path.join(d, "sal")
    py("saliency_map.py", "--ckpt", run, "--data", mc, "--n_samples", "3", "--seed", "5", "--out", sal)
    z = np.load(os.path.join(sal, "saliency_add_local_2.npz"))
    assert z["saliency"].shape[1] == 1324 and np.all(np.diff(z["sample_indices"]) > 0)
    py("plot_saliency.py", "--input", sal, "--out", os.path.join(d, "salfig"), "--max_samples", "1")
    assert os.path.exists(os.path.join(d, "salfig", "index.html"))
    for r in (run, base):
        py("dump_features.py", "--ckpt", r, "--data", mc, "--splits", "clean,add_local_2,jitter_1")
    assert len(glob.glob(os.path.join(run, "features", "features_*.npz"))) == 3
    py("plot_tsne.py", "--static", base, "--adagam", run, "--corruption", "add_local_2", "--out", os.path.join(d, "t.png"))
    from tests.fake_data import make_fake_result
    for i, (t, g, a) in enumerate([("base", "static", "none"), ("adagam", "adagam", "none"),
                                   ("base_aug", "static", "corrupt"), ("adagam_aug", "adagam", "corrupt")]):
        make_fake_result(os.path.join(d, "fr", f"pointMLPElite-{t}-42"), t, g, a, i)
    out = py("plot_figures.py", "--root", os.path.join(d, "fr"), "--data", mc, "--out", os.path.join(d, "figs"))
    for f in ["oa_vs_severity.png", "ce_per_corruption.png", "dalpha_heatmap.png", "corruption_gallery.png"]:
        assert os.path.exists(os.path.join(d, "figs", f)), f


# ------------------------------------------------------------------------------ docs
FORBIDDEN = [r"15 corruption", r"12\.6\s*M", r"15\.8\s*G", r"94\.5%\s*(→|->|to)\s*~?55", r"zero-shot.*ScanObjectNN",
             r"mCE\s*≤\s*78", r"68\s*[–-]\s*72\s*%"]


def test_docs_honest():
    for name in ["README.md", "ANALYSIS.md", os.path.join("docs", "RELATED_WORK.md")]:
        path = os.path.join(ROOT, name)
        assert os.path.exists(path), name
        text = open(path, encoding="utf-8").read()
        for pat in FORBIDDEN:
            hits = [m.group(0) for m in re.finditer(pat, text, flags=re.I)]
            # ANALYSIS.md may quote the old claims only inside its "corrections" table
            if hits and name == "ANALYSIS.md":
                continue
            assert not hits, f"{name}: mid-term claim {hits}"
    readme = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
    assert "TBD (needs GPU run)" in readme and "AdaGAM-3D.git" in readme and "13.4167M" in readme
    rw = open(os.path.join(ROOT, "docs", "RELATED_WORK.md"), encoding="utf-8").read()
    for w in ["FiLM", "Conditional BN", "AdaIN", "Squeeze-and-Excitation", "Self-Modulation", "PointMLP", "ModelNet-C",
              "RPC", "WOLFMix", "PointWOLF", "PointCAT", "RSCNN", "PAConv", "CloudFixer"]:
        assert w in rw, w


# ------------------------------------------------------------------------- hygiene
def test_repo_hygiene():
    for junk in ["images", "pointnet2_ops_lib", "part_segmentation", "classification_ScanObjectNN"]:
        assert not os.path.exists(os.path.join(ROOT, junk)), junk
    assert not glob.glob(os.path.join(ROOT, "**", "*.gif"), recursive=True)
    for f in glob.glob(os.path.join(HERE, "**", "*.py"), recursive=True):
        src = open(f, encoding="utf-8").read()
        assert not re.search(r"os\.system\(|['\"](wget|unzip|rm -rf|mv) ", src), f"POSIX shell call in {f}"
    if os.path.isdir(os.path.join(ROOT, ".git")) and shutil.which("git"):
        log = subprocess.run(["git", "log", "--all", "--format=%an|%ae|%cn|%B%x00"], cwd=ROOT,
                             capture_output=True, text=True).stdout
        for entry in filter(None, (e.strip() for e in log.split("\x00"))):
            author = entry.split("|")[0]
            assert author == "ankur777jinn", f"commit by {author}"
            assert "co-authored-by" not in entry.lower(), "co-author trailer found"
        ignored = subprocess.run(["git", "ls-files", "--others", "--ignored", "--exclude-standard", "--", "*.py"],
                                 cwd=ROOT, capture_output=True, text=True).stdout.split()
        ignored = [f for f in ignored if "__pycache__" not in f]
        assert not ignored, f".gitignore hides source files: {ignored}"


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    order = ["test_smoke_suite", "test_help_lists_flags", "test_runner_order_and_shell", "test_audit_numbers",
             "test_train_resume_eval_summarize", "test_probe_saliency_tsne_figures", "test_docs_honest", "test_repo_hygiene"]
    tests.sort(key=lambda t: order.index(t.__name__) if t.__name__ in order else 99)
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # report and continue
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    if _TMP:
        shutil.rmtree(_TMP, ignore_errors=True)
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
