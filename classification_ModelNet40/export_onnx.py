"""
Export trained PointMLP / AdaGAM checkpoints to ONNX for in-browser inference (web demo).

Farthest-point sampling and k-NN only look at xyz, so they are computed outside the network
(in JavaScript on the web page, `grouping_indices` here) and passed in as inputs. The exported
graph is the rest of the model, works for any number of points N, and has one extra input
`adapt` (1 = AdaGAM as trained, 0 = adaptation switched off, i.e. delta-alpha = delta-beta = 0).

    python export_onnx.py --ckpt checkpoints/pointMLP-adagam-42 --out web/models/adagam.onnx
    python export_onnx.py --ckpt checkpoints/pointMLP-base-42   --out web/models/pointmlp.onnx
    python export_onnx.py --selftest            # untrained model, checks ONNX == PyTorch
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import pointmlp as pm

K = 24


# ---------------------------------------------------------------- grouping (mirrored in JS)
def fps_np(xyz, npoint):
    """Farthest point sampling starting at point 0 (the web page uses the same rule)."""
    n = xyz.shape[0]
    idx = np.zeros(npoint, dtype=np.int64)
    dist = np.full(n, 1e10, dtype=np.float32)
    far = 0
    for i in range(npoint):
        idx[i] = far
        d = ((xyz - xyz[far]) ** 2).sum(-1)
        dist = np.minimum(dist, d)
        far = int(dist.argmax())
    return idx


def knn_np(xyz, centers, k=K):
    d = ((centers[:, None, :] - xyz[None, :, :]) ** 2).sum(-1)
    return np.argpartition(d, k - 1, axis=1)[:, :k].astype(np.int64)


def grouping_indices(xyz):
    """xyz [N,3] float32 -> list of 4 (fps [S], knn [S,K]) pairs, S = N//2, N//4, ..."""
    out, cur = [], xyz
    for _ in range(4):
        s = cur.shape[0] // 2
        f = fps_np(cur, s)
        kn = knn_np(cur, cur[f])
        out.append((f, kn))
        cur = cur[f]
    return out


# ---------------------------------------------------------------- export wrapper (batch 1)
class WebNet(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net.eval()

    def forward(self, pts, fps1, knn1, fps2, knn2, fps3, knn3, fps4, knn4, adapt):
        net = self.net
        xyz = pts[0]                                              # [N,3]
        x = net.embedding(pts.permute(0, 2, 1))                   # [1,D,N]
        feats = x[0].transpose(0, 1)                              # [N,D]
        idx = [(fps1, knn1), (fps2, knn2), (fps3, knn3), (fps4, knn4)]
        for i in range(4):
            g = net.local_grouper_list[i]
            fps, knn = idx[i]
            S = fps.shape[0]
            new_xyz = torch.index_select(xyz, 0, fps)             # [S,3]
            new_pts = torch.index_select(feats, 0, fps)           # [S,D]
            D = feats.shape[1]
            grouped = torch.index_select(feats, 0, knn.reshape(-1)).reshape(S, K, D)
            centered = grouped - new_pts.unsqueeze(1)
            flat = centered.reshape(-1)
            n = flat.shape[0]
            std = torch.sqrt(((flat - flat.mean()) ** 2).sum() / (n - 1))   # torch.std (unbiased)
            xn = centered / (std + 1e-5)
            if hasattr(g, "adagam"):
                a = g.adagam
                ctx = torch.cat([xn.mean(dim=1), xn.amax(dim=1)], dim=-1)    # [S,2D]
                dab = a.fc2(F.leaky_relu(a.fc1(ctx), 0.2))
                da, db = dab[:, :D], dab[:, D:]
                da = (da * adapt).unsqueeze(1)
                db = (db * adapt).unsqueeze(1)
                out = a.affine_alpha[0] * (1.0 + da) * xn + (a.affine_beta[0] + db)
            else:
                out = g.affine_alpha[0] * xn + g.affine_beta[0]
            new = torch.cat([out, new_pts.unsqueeze(1).expand(S, K, D)], dim=-1)   # [S,K,2D]
            x = net.pre_blocks_list[i](new.unsqueeze(0))          # [1,D',S]
            x = net.pos_blocks_list[i](x)
            xyz = new_xyz
            feats = x[0].transpose(0, 1)
        feat = x.amax(dim=-1)                                     # [1,D]
        return F.softmax(net.classifier(feat), dim=-1)


def inputs_for(xyz, adapt=1.0):
    ins = [torch.from_numpy(xyz[None].astype(np.float32))]
    for f, kn in grouping_indices(xyz):
        ins += [torch.from_numpy(f.astype(np.int32)), torch.from_numpy(kn.astype(np.int32))]
    ins.append(torch.tensor([adapt], dtype=torch.float32))
    return ins


NAMES = ["points", "fps1", "knn1", "fps2", "knn2", "fps3", "knn3", "fps4", "knn4", "adapt"]


def export(net, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    wn = WebNet(net)
    xyz = np.random.RandomState(0).randn(1024, 3).astype(np.float32) * 0.4
    dyn = {"points": {1: "N"}}
    for i in range(1, 5):
        dyn[f"fps{i}"] = {0: f"S{i}"}
        dyn[f"knn{i}"] = {0: f"S{i}"}
    with torch.no_grad():
        torch.onnx.export(wn, tuple(inputs_for(xyz)), path, input_names=NAMES, output_names=["probs"],
                          dynamic_axes=dyn, opset_version=17, do_constant_folding=True, dynamo=False)
    return wn


def check(net, path, clouds, tol=1e-3):
    """ONNX vs the export wrapper vs the ORIGINAL model (with its FPS forced to start at point 0)."""
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    wn = WebNet(net)
    orig_fps = pm.farthest_point_sample

    def fps_from0(xyz, npoint):
        return torch.from_numpy(fps_np(xyz[0].numpy(), npoint))[None]
    worst = 0.0
    for xyz in clouds:
        for adapt in ([1.0, 0.0] if hasattr(net.local_grouper_list[0], "adagam") else [1.0]):
            ins = inputs_for(xyz, adapt)
            have = {i.name for i in sess.get_inputs()}       # static PointMLP has no `adapt`
            o = sess.run(None, {n: t.numpy() for n, t in zip(NAMES, ins) if n in have})[0]
            with torch.no_grad():
                w = wn(*ins).numpy()
                pm.farthest_point_sample = fps_from0
                for m in net.modules():
                    if m.__class__.__name__ == "AdaGAM":
                        m.disable_adaptation = adapt == 0.0
                ref = F.softmax(net(torch.from_numpy(xyz[None]).permute(0, 2, 1)), -1).numpy()
                pm.farthest_point_sample = orig_fps
                for m in net.modules():
                    if m.__class__.__name__ == "AdaGAM":
                        m.disable_adaptation = False
            worst = max(worst, float(np.abs(o - w).max()), float(np.abs(o - ref).max()))
            assert o.argmax() == ref.argmax(), "prediction mismatch"
    print(f"max |prob difference| ONNX vs PyTorch over {len(clouds)} clouds: {worst:.2e}")
    assert worst < tol, worst


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt")
    p.add_argument("--out")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    torch.set_grad_enabled(False)
    rng = np.random.RandomState(1)
    clouds = [rng.randn(n, 3).astype(np.float32) * 0.4 for n in (1024, 1074, 524, 256, 1524)]
    if a.selftest:
        for gam in ("static", "adagam"):
            torch.manual_seed(0)
            net = models.pointMLP(gam=gam, gam_kwargs=dict(reduction=8, use_std=False, norm="pointmlp")).eval()
            for m in net.modules():          # non-trivial weights/statistics
                if isinstance(m, (nn.BatchNorm1d,)):
                    m.running_mean.normal_(0, 0.1); m.running_var.uniform_(0.5, 1.5)
                if m.__class__.__name__ == "AdaGAM":
                    nn.init.normal_(m.fc2.weight, std=0.05)
            path = os.path.join("web_selftest", f"{gam}.onnx")
            export(net, path)
            check(net, path, clouds)
            print(gam, "OK", f"{os.path.getsize(path) / 1e6:.1f} MB")
        return
    from eval_modelnet_c import load_model
    net, _ = load_model(a.ckpt, "cpu")
    net.eval()
    export(net, a.out)
    check(net, a.out, clouds)
    print("wrote", a.out, f"{os.path.getsize(a.out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
