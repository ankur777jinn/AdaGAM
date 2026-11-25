"""CPU sanity checks. Run this first on any new machine and after every change in models/.

    python smoke_test.py            # must end with "All smoke tests passed."
"""
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import models  # noqa: E402
from corruptions import CORRUPTIONS, apply_corruption, output_size  # noqa: E402
from models.adagam import AdaGAM  # noqa: E402

torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


def _pair(name, **kw):
    torch.manual_seed(0)
    static = models.build_model(name, gam="static", **kw).eval()
    ada = models.build_model(name, gam="adagam", reduction=8, **kw).eval()
    missing, unexpected = ada.load_state_dict(static.state_dict(), strict=False)
    assert not unexpected and all(".adagam." in k for k in missing), (missing, unexpected)
    return static, ada


def test_init_parity():
    for name in ("pointMLPElite", "pointMLP"):
        static, ada = _pair(name)
        x = torch.rand(2, 3, 1024) * 2 - 1
        with torch.no_grad():
            d = (static(x) - ada(x)).abs().max().item()
        assert d == 0.0, f"{name}: untrained AdaGAM differs from static by {d}"
        print(f"  [1] {name}: static vs untrained AdaGAM max diff = {d:.2e}")
    static, ada = _pair("pointMLPElite", gam_norm="patch")
    with torch.no_grad():
        x = torch.rand(2, 3, 1024)
        assert (static(x) - ada(x)).abs().max().item() == 0.0
    ada_b = models.build_model("pointMLPElite", gam="adagam", bound=True).eval()
    ada_b.load_state_dict(static.state_dict(), strict=False)
    print("  [1] gam_norm=patch parity OK; tanh-bounded variant builds")


def test_permutation_and_determinism():
    torch.manual_seed(1)
    m = AdaGAM(64, reduction=8, use_std=True, use_min=True)
    torch.nn.init.normal_(m.fc2.weight, std=0.1)
    x = torch.randn(2, 16, 24, 64)
    perm = torch.randperm(24)
    a1, b1, _ = m(x)
    a2, b2, _ = m(x[:, :, perm])
    assert torch.allclose(a1, a2, atol=1e-6) and torch.allclose(b1, b2, atol=1e-6)
    _, ada = _pair("pointMLPElite")
    for mod in ada.adagam_modules():
        torch.nn.init.normal_(mod.fc2.weight, std=0.05)
    x = torch.rand(3, 3, 1024)
    with torch.no_grad():
        o1, o2 = ada(x), ada(x)
    assert torch.equal(o1, o2), "two eval forwards differ (FPS not deterministic in eval?)"
    print("  [2] permutation invariance over K OK; eval forward is bit-identical across calls")


def test_variable_n():
    for gam in ("static", "adagam"):
        net = models.build_model("pointMLPElite", gam=gam).eval()
        assert net.min_points == 192, net.min_points
        with torch.no_grad():
            for n in (192, 256, 1024, 1524):
                assert net(torch.rand(2, 3, n)).shape == (2, 40)
        try:
            net(torch.rand(1, 3, 150))
            raise AssertionError("expected ValueError for a 150-point cloud")
        except ValueError:
            pass
    print("  [3] variable N OK (192, 256, 1024, 1524 points; <192 rejected with a clear error)")


def test_corruptions():
    rng = np.random.default_rng(0)
    pc = rng.uniform(-1, 1, (1024, 3)).astype(np.float32)
    for c in CORRUPTIONS:
        for lvl in range(5):
            out = apply_corruption(pc, c, lvl, np.random.default_rng(lvl))
            assert out.shape == (output_size(c, lvl), 3), (c, lvl, out.shape)
            assert np.isfinite(out).all()
    assert apply_corruption(pc[:300], "add_local", 4, rng).shape == (800, 3)      # used to crash for N < 500
    assert apply_corruption(pc, "drop_global", 0, rng).shape == (768, 3)         # alias
    a = apply_corruption(pc, "jitter", 2, np.random.default_rng(7))
    b = apply_corruption(pc, "jitter", 2, np.random.default_rng(7))
    assert np.array_equal(a, b), "seeded corruption not reproducible"
    from data import CorruptCollate
    batch = [(rng.uniform(-1, 1, (1024, 3)).astype(np.float32), np.int64(i)) for i in range(4)]
    for c in CORRUPTIONS:
        col = CorruptCollate([c], max_severity=4)
        clean, corr, lab, name, lvl = col(batch)
        assert clean.shape == (4, 1024, 3) and corr.shape == (4, output_size(c, lvl), 3) and name == c
    print("  [4] 7 corruptions x 5 severities OK (sizes match ModelNet-C); pair collate OK; seeded RNG reproducible")


def test_gradients_and_counterfactual():
    torch.manual_seed(0)
    static, ada = _pair("pointMLPElite")
    ada.train()
    opt = torch.optim.SGD(ada.parameters(), lr=0.1, momentum=0.9)
    x, y = torch.rand(4, 3, 1024), torch.randint(0, 40, (4,))
    for step in range(2):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(ada(x), y).backward()
        g1 = sum(m.fc1.weight.grad.abs().sum().item() for m in ada.adagam_modules())
        g2 = sum(m.fc2.weight.grad.abs().sum().item() for m in ada.adagam_modules())
        opt.step()
    assert g2 > 0 and g1 > 0, f"gradient did not reach AdaGAM (fc1 {g1}, fc2 {g2})"
    ada.eval()
    static.load_state_dict({k: v for k, v in ada.state_dict().items() if ".adagam." not in k})
    static.eval()
    xe = torch.rand(2, 3, 1024)
    with torch.no_grad():
        on = ada(xe)
        off = ada(xe, adapt=False)
        for m in ada.adagam_modules():
            m.disable_adaptation = True
        off_flag = ada(xe)
        for m in ada.adagam_modules():
            m.disable_adaptation = False
        ref = static(xe)
    assert (off - ref).abs().max().item() == 0.0 and (off_flag - ref).abs().max().item() == 0.0
    assert (on - ref).abs().max().item() > 0, "trained AdaGAM should differ from its counterfactual"
    print(f"  [5] gradients reach fc1 ({g1:.2e}) and fc2 ({g2:.2e}); adapt=False == static output exactly")


def test_eval_pipeline():
    from tests.fake_data import make_fake_result, make_modelnet_c
    from eval_modelnet_c import evaluate_run
    import summarize
    tmp = tempfile.mkdtemp(prefix="adagam_smoke_")
    try:
        mc = make_modelnet_c(tmp, n=4)
        run = os.path.join(tmp, "ckpt", "pointMLPElite-adagam-42")
        os.makedirs(run)
        net = models.build_model("pointMLPElite", gam="adagam")
        torch.save({"net": net.state_dict(), "epoch": 0}, os.path.join(run, "last_checkpoint.pth"))
        with open(os.path.join(run, "args.json"), "w") as f:
            json.dump({"model": "pointMLPElite", "gam": "adagam", "reduction": 8, "aug": "none"}, f)
        res = evaluate_run(run, mc, "last", "cpu", batch_size=4, verbose=False)
        assert len(res["oa"]["scale"]) == 5 and res["adagam_dalpha"] is not None and "mCE" in res
        res2 = evaluate_run(run, mc, "last", "cpu", batch_size=4, verbose=False)   # resumes from cache
        assert res2["mCE"] == res["mCE"]
        make_fake_result(os.path.join(tmp, "ckpt", "pointMLPElite-base_aug-42"), "base_aug", "static", "corrupt", 1)
        rows = summarize.collect(os.path.join(tmp, "ckpt"))
        summarize.write(rows, os.path.join(tmp, "results.md"), os.path.join(tmp, "results.csv"))
        md = open(os.path.join(tmp, "results.md"), encoding="utf-8").read()
        assert "Seen OA" in md and "Unseen OA" in md and "not leaderboard-comparable" in md
        print(f"  [6] ModelNet-C eval on fake data OK (mCE {res['mCE']:.3f}, resumable); results.md has seen/unseen columns")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    t0 = time.time()
    tests = [test_init_parity, test_permutation_and_determinism, test_variable_n, test_corruptions,
             test_gradients_and_counterfactual, test_eval_pipeline]
    for t in tests:
        print(f"- {t.__name__}")
        t()
    print(f"All smoke tests passed. ({time.time() - t0:.1f}s, torch {torch.__version__}, "
          f"FPS: {'pointnet2_ops' if models.HAS_POINTNET2_OPS else 'pure PyTorch'})")


if __name__ == "__main__":
    main()
