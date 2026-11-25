"""
CPU sanity checks, no dataset needed:  python smoke_test.py
1. AdaGAM == static GAM at initialisation (zero-init fc2)
2. AdaGAM is permutation invariant over the K neighbours
3. Variable point counts (ModelNet-C drop/add) run
4. All 7 training corruptions x 5 severities keep shape [N,3]
5. One corrupt+consistency training step backpropagates into AdaGAM
6. eval_modelnet_c.py runs end-to-end on fake ModelNet-C files
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import h5py
import numpy as np
import torch

import models
from models.adagam import AdaGAM
from corruptions import ALL_CORRUPTIONS, _FUNCS
from main import consistency_loss
from utils import cal_loss

torch.manual_seed(0)
np.random.seed(0)
MODEL = "pointMLPElite"


def check(name, ok):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        sys.exit(1)


# 1. identity at init --------------------------------------------------------------
base = models.__dict__[MODEL]().eval()
ada = models.__dict__[MODEL](gam="adagam").eval()
sd = base.state_dict()
mapped = {}
for k, v in sd.items():
    k2 = k.replace(".affine_alpha", ".adagam.affine_alpha").replace(".affine_beta", ".adagam.affine_beta")
    mapped[k2] = v
missing, unexpected = ada.load_state_dict(mapped, strict=False)
check("only AdaGAM MLP weights are new", all(".adagam.fc" in m for m in missing) and not unexpected)
x = torch.rand(2, 3, 1024)
with torch.no_grad():
    torch.manual_seed(1); yb = base(x)
    torch.manual_seed(1); ya = ada(x)
check(f"AdaGAM == static GAM at init (max diff {(yb - ya).abs().max():.2e})", torch.allclose(yb, ya, atol=1e-5))

# 2. permutation invariance ----------------------------------------------------------
m = AdaGAM(32)
torch.nn.init.normal_(m.fc2.weight, std=0.1)
g = torch.randn(2, 8, 24, 32)
a = torch.randn(2, 8, 1, 32)
perm = torch.randperm(24)
check("permutation invariant over K", torch.allclose(m(g, a)[:, :, perm], m(g[:, :, perm], a), atol=1e-5))
mp = AdaGAM(32, norm="patch", use_std=True)
torch.nn.init.normal_(mp.fc2.weight, std=0.1)
check("patch-norm + std variant runs & is invariant",
      torch.allclose(mp(g, a)[:, :, perm], mp(g[:, :, perm], a), atol=1e-5))

# 3. variable N --------------------------------------------------------------------
with torch.no_grad():
    for n in (256, 524, 1124):
        out = ada(torch.rand(2, 3, n))
        check(f"forward with N={n} -> {tuple(out.shape)}", out.shape == (2, 40))

# 4. corruptions -------------------------------------------------------------------
pc = np.random.uniform(-1, 1, (1024, 3)).astype("float32")
pc /= np.linalg.norm(pc, axis=1).max()
ok = all(_FUNCS[c](pc.copy(), s).shape == (1024, 3) for c in ALL_CORRUPTIONS for s in range(5))
check("7 corruptions x 5 severities keep shape", ok)

# 5. training step -----------------------------------------------------------------
net = models.__dict__[MODEL](gam="adagam").train()
opt = torch.optim.SGD(net.parameters(), lr=0.1)
clean = torch.rand(4, 3, 1024)
corrupt = torch.from_numpy(np.stack([_FUNCS["jitter"](c.T.numpy().copy(), 4) for c in clean])).permute(0, 2, 1)
label = torch.randint(0, 40, (4,))
for kind in ("feat", "jsd"):
    opt.zero_grad()
    logits, feat = net(torch.cat([clean, corrupt]), return_feat=True)
    loss = cal_loss(logits[:4], label) + cal_loss(logits[4:], label) + \
        1.0 * consistency_loss(kind, logits[:4], logits[4:], feat[:4], feat[4:])
    loss.backward()
    gnorm = sum(p.grad.abs().sum().item() for n_, p in net.named_parameters() if "adagam.fc2" in n_)
    check(f"consistency={kind}: loss={loss.item():.3f}, grad into AdaGAM fc2={gnorm:.3e}",
          np.isfinite(loss.item()) and gnorm > 0)
    opt.step()

# 6. eval script on fake ModelNet-C ----------------------------------------------------
tmp = tempfile.mkdtemp()
try:
    ck = os.path.join(tmp, "ckpt")
    dd = os.path.join(tmp, "modelnet_c")
    os.makedirs(ck); os.makedirs(dd)
    json.dump(dict(model=MODEL, gam="adagam", reduction=4, use_std=False, gam_norm="pointmlp"),
              open(os.path.join(ck, "args.json"), "w"))
    torch.save({"net": {"module." + k: v for k, v in net.state_dict().items()}},
               os.path.join(ck, "last_checkpoint.pth"))
    sizes = {"dropout_global": 256, "dropout_local": 700, "add_global": 1074, "add_local": 1324}
    names = ["clean"] + [f"{c}_{l}" for c in ["scale", "jitter", "rotate", "dropout_global",
                                              "dropout_local", "add_global", "add_local"] for l in range(5)]
    for nm in names:
        n = sizes.get(nm.rsplit("_", 1)[0], 1024)
        with h5py.File(os.path.join(dd, nm + ".h5"), "w") as f:
            f["data"] = np.random.uniform(-1, 1, (6, n, 3)).astype("float32")
            f["label"] = np.random.randint(0, 40, (6, 1))
    r = subprocess.run([sys.executable, "eval_modelnet_c.py", "--ckpt", ck, "--data", dd, "--bs", "4"],
                       capture_output=True, text=True)
    print(r.stdout[-600:], r.stderr[-1500:])
    res = json.load(open(os.path.join(ck, "modelnet_c.json")))
    check(f"eval_modelnet_c.py end-to-end (mCE={res['mCE']:.3f})", r.returncode == 0 and "mCE" in res)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\nAll smoke tests passed.")
