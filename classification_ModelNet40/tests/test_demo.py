"""Demo checks: starts with NO checkpoint and NO dataset, drives every control, checks upload errors.

    python -m pytest classification_ModelNet40/tests/test_demo.py -q
    python classification_ModelNet40/tests/test_demo.py
"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

from demo_app import CORR_CHOICES, DemoState, build_app, run  # noqa: E402
from inference_engine import InferenceEngine  # noqa: E402
from shapes import make_shape, shape_names  # noqa: E402

_STATE = None


def state():
    global _STATE
    if _STATE is None:
        _STATE = DemoState(InferenceEngine(device="cpu"), data_dir=None)     # no .pth, no .h5
    return _STATE


def test_starts_without_checkpoint_or_data():
    s = state()
    assert s.engine.demo_weights and s.data_dir is None
    app = build_app(s)
    assert app is not None


def test_shapes_distinct_and_labelled():
    names = shape_names()
    assert len(names) >= 8 and len(set(names)) == len(names)
    clouds = [make_shape(n, 1024, 0) for n in names]
    for n, c in zip(names, clouds):
        assert c.shape == (1024, 3) and np.isfinite(c).all() and np.linalg.norm(c, axis=1).max() <= 1.0 + 1e-5, n
    # shapes really differ (sorted radius profiles are not identical)
    prof = [np.sort(np.linalg.norm(c, axis=1)) for c in clouds]
    for i in range(len(prof)):
        for j in range(i + 1, len(prof)):
            assert np.abs(prof[i] - prof[j]).mean() > 1e-3, (names[i], names[j])


def test_every_corruption_and_severity():
    s = state()
    for c in CORR_CHOICES:
        for sev in range(1, 6):
            fig, st, ad, tele = run(s, "Procedural shape", "chair", 0, None, c, sev, True, 0)
            assert fig is not None and len(st) == 5 and len(ad) == 5, (c, sev, tele)


def test_toggle_keeps_cloud_and_counterfactual_matches_static():
    s = state()
    on = run(s, "Procedural shape", "table", 0, None, "dropout_global", 5, True, 3)
    off = run(s, "Procedural shape", "table", 0, None, "dropout_global", 5, False, 3)
    for axis in ("x", "y", "z"):
        assert np.array_equal(np.asarray(getattr(on[0].data[0], axis)), np.asarray(getattr(off[0].data[0], axis)))
    assert len(on[0].data[0].x) == 256                       # variable N survives (75% dropped)
    assert off[2] == off[1]                                  # adaptation off == static (same backbone here)


def test_uploads_and_friendly_errors():
    s = state()
    d = tempfile.mkdtemp()
    pc = make_shape("cup", 2000, 1)
    np.save(os.path.join(d, "ok.npy"), pc)
    np.savetxt(os.path.join(d, "ok.xyz"), pc)
    with open(os.path.join(d, "ok.ply"), "wb") as f:
        f.write(f"ply\nformat binary_little_endian 1.0\nelement vertex {len(pc)}\nproperty float x\n"
                f"property float y\nproperty float z\nend_header\n".encode() + pc.astype("<f4").tobytes())
    for fn in ("ok.npy", "ok.xyz", "ok.ply"):
        r = run(s, "Upload", None, 0, os.path.join(d, fn), "jitter", 2, True, 0)
        assert r[0] is not None, (fn, r[3])
    open(os.path.join(d, "bad.npy"), "wb").write(b"not numpy")
    np.save(os.path.join(d, "nan.npy"), np.full((64, 3), np.nan))
    np.save(os.path.join(d, "shape.npy"), np.zeros((5, 5, 5, 5)))
    open(os.path.join(d, "evil.exe"), "w").write("x")
    for fn in ("bad.npy", "nan.npy", "shape.npy", "evil.exe", None):
        r = run(s, "Upload", None, 0, None if fn is None else os.path.join(d, fn), "clean", 1, True, 0)
        assert r[0] is None and r[3].startswith("⚠️"), (fn, r[3])
    r = run(s, "ModelNet-C test object", None, 0, None, "clean", 1, True, 0)
    assert r[0] is None and "--data" in r[3]


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print("PASS", name)
            except Exception as e:
                fails += 1
                print("FAIL", name, e)
    sys.exit(1 if fails else 0)
