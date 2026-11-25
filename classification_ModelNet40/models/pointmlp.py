"""PointMLP backbone (Ma et al., ICLR 2022) with an optional AdaGAM geometric affine.

Re-implemented from the official design (https://github.com/ma-xu/pointMLP-pytorch) with:
  * gam="static" (original) or gam="adagam" (adaptive affine, identical to static at init);
  * the number of anchors at each stage scales with the input size (S = N // reducer), so the
    variable-size ModelNet-C clouds (256 .. 1524 points) run without resampling;
  * a vectorised pure-PyTorch farthest point sampling (FPS) fallback when the CUDA op
    `pointnet2_ops` is not installed (it is hard to build on Windows). It starts at point 0 in eval
    mode, like the CUDA op, so evaluation is deterministic;
  * kNN / normalisation run in float32 even under autocast (AMP-safe);
  * forward(x, return_feat=True) returns (logits, global_feature);
    forward(x, return_aux=True) returns (logits, global_feature, aux_per_stage) with no module
    state mutated (thread-safe for the demo); adapt=False switches AdaGAM off on the same weights.
"""
import contextlib
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adagam import AdaGAM, pool_context

try:  # optional CUDA FPS
    from pointnet2_ops import pointnet2_utils as _p2u  # type: ignore
    HAS_POINTNET2_OPS = True
except Exception:  # pragma: no cover - depends on the machine
    _p2u = None
    HAS_POINTNET2_OPS = False

FPS_BACKEND = "auto"   # "auto" | "torch" | "cuda" ; set via set_fps_backend()


def set_fps_backend(name: str) -> None:
    global FPS_BACKEND
    if name not in ("auto", "torch", "cuda"):
        raise ValueError(f"unknown FPS backend {name}")
    if name == "cuda" and not HAS_POINTNET2_OPS:
        raise RuntimeError("fps backend 'cuda' requested but pointnet2_ops is not installed")
    FPS_BACKEND = name


def get_activation(activation: str) -> nn.Module:
    a = activation.lower()
    if a == "gelu":
        return nn.GELU()
    if a == "rrelu":
        return nn.RReLU(inplace=True)
    if a == "selu":
        return nn.SELU(inplace=True)
    if a == "silu":
        return nn.SiLU(inplace=True)
    if a == "hardswish":
        return nn.Hardswish(inplace=True)
    if a == "leakyrelu":
        return nn.LeakyReLU(inplace=True)
    return nn.ReLU(inplace=True)


def _no_autocast(device_type: str):
    try:
        return torch.autocast(device_type=device_type, enabled=False)
    except Exception:  # pragma: no cover - very old torch / unsupported device
        return contextlib.nullcontext()


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean distance. src [B, N, C], dst [B, M, C] -> [B, N, M]."""
    if src.dim() != 3 or dst.dim() != 3 or src.shape[0] != dst.shape[0] or src.shape[-1] != dst.shape[-1]:
        raise ValueError(f"square_distance shape mismatch: {tuple(src.shape)} vs {tuple(dst.shape)}")
    dist = -2 * torch.matmul(src, dst.transpose(1, 2))
    dist = dist + torch.sum(src ** 2, -1).unsqueeze(-1)
    dist = dist + torch.sum(dst ** 2, -1).unsqueeze(-2)
    return dist


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """points [B, N, C], idx [B, ...] -> [B, ..., C]."""
    B = points.shape[0]
    view_shape = [B] + [1] * (idx.dim() - 1)
    batch_indices = torch.arange(B, dtype=torch.long, device=points.device).view(view_shape).expand_as(idx)
    return points[batch_indices, idx, :]


def _fps_torch(xyz: torch.Tensor, npoint: int, deterministic: bool) -> torch.Tensor:
    """Vectorised over the batch; the only Python loop is over the `npoint` picks."""
    B, N, _ = xyz.shape
    device = xyz.device
    centroids = torch.empty(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), float("inf"), device=device, dtype=xyz.dtype)
    if deterministic:
        farthest = torch.zeros(B, dtype=torch.long, device=device)
    else:
        farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest].unsqueeze(1)          # [B, 1, 3]
        dist = ((xyz - centroid) ** 2).sum(-1)                         # [B, N]
        distance = torch.minimum(distance, dist)
        farthest = distance.argmax(-1)
    return centroids


def farthest_point_sample(xyz: torch.Tensor, npoint: int, deterministic: bool = False) -> torch.Tensor:
    """xyz [B, N, 3] -> indices [B, npoint]. CUDA op if available (always starts at 0)."""
    if xyz.dim() != 3 or xyz.shape[-1] != 3:
        raise ValueError(f"farthest_point_sample expects [B, N, 3], got {tuple(xyz.shape)}")
    N = xyz.shape[1]
    if not (1 <= npoint <= N):
        raise ValueError(f"farthest_point_sample: need 1 <= npoint ({npoint}) <= N ({N})")
    use_cuda_op = (HAS_POINTNET2_OPS and xyz.is_cuda and FPS_BACKEND in ("auto", "cuda"))
    if use_cuda_op:
        return _p2u.furthest_point_sample(xyz.float().contiguous(), npoint).long()
    return _fps_torch(xyz.float(), npoint, deterministic)


def knn_point(nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """For each query in new_xyz [B, S, 3] return the indices of its nsample nearest points in xyz [B, N, 3]."""
    N = xyz.shape[1]
    if nsample > N:
        raise ValueError(f"knn_point: k={nsample} > N={N}; input cloud too small for this stage")
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


class LocalGrouper(nn.Module):
    """FPS -> kNN -> normalisation -> (static | adaptive) geometric affine.

    gam_norm="pointmlp" (default): anchor-centred, divided by ONE std per sample (PointMLP).
    gam_norm="patch": mid-term ablation, per-channel / per-neighbourhood normalisation of the
                      anchor-centred features. The AdaGAM context always comes from the
                      PointMLP-normalised features, so it keeps per-patch scale information.
    """

    def __init__(self, channel: int, reduce: int, kneighbors: int, use_xyz: bool = False,
                 normalize: str = "anchor", gam: str = "static", reduction: int = 8,
                 use_std: bool = False, use_min: bool = False, gam_norm: str = "pointmlp",
                 bound: bool = False):
        super().__init__()
        if gam not in ("static", "adagam"):
            raise ValueError(f"gam must be 'static' or 'adagam', got {gam}")
        if gam_norm not in ("pointmlp", "patch"):
            raise ValueError(f"gam_norm must be 'pointmlp' or 'patch', got {gam_norm}")
        if normalize not in ("anchor", "center", None):
            raise ValueError(f"normalize must be 'anchor', 'center' or None, got {normalize}")
        if reduce < 1 or kneighbors < 1:
            raise ValueError("reduce and kneighbors must be >= 1")
        self.reduce, self.kneighbors, self.use_xyz = int(reduce), int(kneighbors), use_xyz
        self.normalize, self.gam, self.gam_norm = normalize, gam, gam_norm
        add_channel = 3 if use_xyz else 0
        self.channels = channel + add_channel
        self.affine_alpha = nn.Parameter(torch.ones([1, 1, 1, self.channels]))
        self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, self.channels]))
        self.use_std, self.use_min = use_std, use_min
        self.adagam = (AdaGAM(self.channels, reduction=reduction, use_std=use_std, use_min=use_min,
                              bound=bound) if gam == "adagam" else None)

    def forward(self, xyz: torch.Tensor, points: torch.Tensor, adapt: bool = True,
                return_aux: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
        """xyz [B, N, 3], points [B, N, D] -> new_xyz [B, S, 3], new_points [B, S, K, 2D(+3)]."""
        if xyz.dim() != 3 or points.dim() != 3 or xyz.shape[:2] != points.shape[:2]:
            raise ValueError(f"LocalGrouper shape mismatch: xyz {tuple(xyz.shape)}, points {tuple(points.shape)}")
        B, N, _ = xyz.shape
        S = N // self.reduce
        if S < 1 or N < self.kneighbors:
            raise ValueError(f"LocalGrouper: N={N} too small (need N >= k={self.kneighbors} and N // {self.reduce} >= 1)")

        with _no_autocast(xyz.device.type):
            xyz = xyz.float().contiguous()
            points = points.float()
            fps_idx = farthest_point_sample(xyz, S, deterministic=not self.training)   # [B, S]
            new_xyz = index_points(xyz, fps_idx)                                        # [B, S, 3]
            new_points = index_points(points, fps_idx)                                  # [B, S, D]
            idx = knn_point(self.kneighbors, xyz, new_xyz)                              # [B, S, K]
            grouped_xyz = index_points(xyz, idx)                                        # [B, S, K, 3]
            grouped = index_points(points, idx)                                         # [B, S, K, D]
            if self.use_xyz:
                grouped = torch.cat([grouped, grouped_xyz], dim=-1)
            anchor_feat = torch.cat([new_points, new_xyz], dim=-1) if self.use_xyz else new_points

            if self.normalize is None:
                x_ctx = grouped
            else:
                if self.normalize == "center":
                    mean = grouped.mean(dim=2, keepdim=True)
                else:  # anchor
                    mean = anchor_feat.unsqueeze(dim=-2)
                centered = grouped - mean
                std = torch.std(centered.reshape(B, -1), dim=-1, keepdim=True).view(B, 1, 1, 1)
                x_ctx = centered / (std + 1e-5)                                        # PointMLP normalisation

            if self.gam_norm == "patch" and self.normalize is not None:
                mu = x_ctx.mean(dim=2, keepdim=True)
                sd = x_ctx.var(dim=2, keepdim=True, unbiased=False).add(1e-5).sqrt()
                x_mod = (x_ctx - mu) / sd
            else:
                x_mod = x_ctx

            aux: Optional[Dict] = None
            if self.adagam is not None:
                d_alpha, d_beta, a_aux = self.adagam(x_ctx, adapt=adapt)
                out = (self.affine_alpha * (1 + d_alpha)) * x_mod + (self.affine_beta + d_beta)
            else:
                a_aux = None
                out = self.affine_alpha * x_mod + self.affine_beta

            if return_aux:
                aux = {"fps_idx": fps_idx, "knn_idx": idx, "new_xyz": new_xyz.detach()}
                if a_aux is not None:
                    aux.update(a_aux)
                else:  # parity features for the static model (same pooling as AdaGAM would use)
                    aux["context"] = pool_context(x_ctx, self.use_std, self.use_min).detach()
                    aux["dalpha_group"] = torch.zeros(B, S, device=xyz.device)
                    aux["dbeta_group"] = torch.zeros(B, S, device=xyz.device)

            # like upstream PointMLP: append the anchor's own features (not its xyz) to every neighbour
            out = torch.cat([out, new_points.view(B, S, 1, -1).expand(-1, -1, self.kneighbors, -1)], dim=-1)
        return new_xyz, out, aux


class ConvBNReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True, activation="relu"):
        super().__init__()
        self.act = get_activation(activation)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act,
        )

    def forward(self, x):
        return self.net(x)


class ConvBNReLURes1D(nn.Module):
    def __init__(self, channel, kernel_size=1, groups=1, res_expansion=1.0, bias=True, activation="relu"):
        super().__init__()
        self.act = get_activation(activation)
        mid = int(channel * res_expansion)
        self.net1 = nn.Sequential(
            nn.Conv1d(channel, mid, kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(mid),
            self.act,
        )
        if groups > 1:
            self.net2 = nn.Sequential(
                nn.Conv1d(mid, channel, kernel_size=kernel_size, groups=groups, bias=bias),
                nn.BatchNorm1d(channel),
                self.act,
                nn.Conv1d(channel, channel, kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel),
            )
        else:
            self.net2 = nn.Sequential(
                nn.Conv1d(mid, channel, kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel),
            )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)


class PreExtraction(nn.Module):
    def __init__(self, channels, out_channels, blocks=1, groups=1, res_expansion=1, bias=True,
                 activation="relu", use_xyz=True):
        super().__init__()
        in_channels = 3 + 2 * channels if use_xyz else 2 * channels
        self.transfer = ConvBNReLU1D(in_channels, out_channels, bias=bias, activation=activation)
        self.operation = nn.Sequential(*[
            ConvBNReLURes1D(out_channels, groups=groups, res_expansion=res_expansion, bias=bias, activation=activation)
            for _ in range(blocks)])

    def forward(self, x):
        b, n, s, d = x.size()                       # [B, S, K, D']
        x = x.permute(0, 1, 3, 2).reshape(-1, d, s)
        x = self.transfer(x)
        batch_size, _, _ = x.size()
        x = self.operation(x)                        # [B*S, D, K]
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        return x.reshape(b, n, -1).permute(0, 2, 1)  # [B, D, S]


class PosExtraction(nn.Module):
    def __init__(self, channels, blocks=1, groups=1, res_expansion=1, bias=True, activation="relu"):
        super().__init__()
        self.operation = nn.Sequential(*[
            ConvBNReLURes1D(channels, groups=groups, res_expansion=res_expansion, bias=bias, activation=activation)
            for _ in range(blocks)])

    def forward(self, x):
        return self.operation(x)


class Model(nn.Module):
    def __init__(self, points=1024, class_num=40, embed_dim=64, groups=1, res_expansion=1.0,
                 activation="relu", bias=True, use_xyz=True, normalize="center",
                 dim_expansion=(2, 2, 2, 2), pre_blocks=(2, 2, 2, 2), pos_blocks=(2, 2, 2, 2),
                 k_neighbors=(32, 32, 32, 32), reducers=(2, 2, 2, 2),
                 gam="static", reduction=8, use_std=False, use_min=False, gam_norm="pointmlp",
                 bound=False, **kwargs):
        super().__init__()
        n = len(pre_blocks)
        if not (len(k_neighbors) == len(reducers) == len(pos_blocks) == len(dim_expansion) == n):
            raise ValueError("stage lists must have equal length")
        self.stages = n
        self.class_num = class_num
        self.points = points
        self.gam = gam
        self.k_neighbors, self.reducers = list(k_neighbors), list(reducers)
        self.embedding = ConvBNReLU1D(3, embed_dim, bias=bias, activation=activation)
        self.local_grouper_list = nn.ModuleList()
        self.pre_blocks_list = nn.ModuleList()
        self.pos_blocks_list = nn.ModuleList()
        last_channel = embed_dim
        for i in range(n):
            out_channel = last_channel * dim_expansion[i]
            self.local_grouper_list.append(LocalGrouper(
                last_channel, reducers[i], k_neighbors[i], use_xyz, normalize, gam=gam,
                reduction=reduction, use_std=use_std, use_min=use_min, gam_norm=gam_norm, bound=bound))
            self.pre_blocks_list.append(PreExtraction(
                last_channel, out_channel, pre_blocks[i], groups=groups, res_expansion=res_expansion,
                bias=bias, activation=activation, use_xyz=use_xyz))
            self.pos_blocks_list.append(PosExtraction(
                out_channel, pos_blocks[i], groups=groups, res_expansion=res_expansion, bias=bias,
                activation=activation))
            last_channel = out_channel
        self.feat_dim = last_channel
        self.act = get_activation(activation)
        self.classifier = nn.Sequential(
            nn.Linear(last_channel, 512), nn.BatchNorm1d(512), self.act, nn.Dropout(0.5),
            nn.Linear(512, 256), nn.BatchNorm1d(256), self.act, nn.Dropout(0.5),
            nn.Linear(256, self.class_num))

    @property
    def min_points(self) -> int:
        """Smallest cloud the network accepts: every stage needs at least k input points."""
        need, prod = 1, 1
        for k, r in zip(self.k_neighbors, self.reducers):
            need = max(need, k * prod, r * prod)
            prod *= r
        return need

    def adagam_modules(self) -> List[AdaGAM]:
        return [g.adagam for g in self.local_grouper_list if g.adagam is not None]

    def forward(self, x: torch.Tensor, return_feat: bool = False, adapt: bool = True,
                return_aux: bool = False):
        """x: [B, 3, N]. Returns logits; (logits, feat) if return_feat;
        (logits, feat, aux_list) if return_aux."""
        if x.dim() != 3 or x.shape[1] != 3:
            raise ValueError(f"Model expects [B, 3, N], got {tuple(x.shape)}")
        if x.shape[2] < self.min_points:
            raise ValueError(f"input has {x.shape[2]} points; this model needs >= {self.min_points}")
        xyz = x.permute(0, 2, 1)
        x = self.embedding(x)
        aux_list = []
        for i in range(self.stages):
            xyz, x, aux = self.local_grouper_list[i](xyz, x.permute(0, 2, 1), adapt=adapt, return_aux=return_aux)
            x = self.pre_blocks_list[i](x)
            x = self.pos_blocks_list[i](x)
            if return_aux:
                aux_list.append(aux)
        feat = F.adaptive_max_pool1d(x, 1).squeeze(dim=-1)
        logits = self.classifier(feat)
        if return_aux:
            return logits, feat, aux_list
        if return_feat:
            return logits, feat
        return logits


def pointMLP(num_classes=40, **kwargs) -> Model:
    return Model(points=1024, class_num=num_classes, embed_dim=64, groups=1, res_expansion=1.0,
                 activation="relu", bias=False, use_xyz=False, normalize="anchor",
                 dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[24, 24, 24, 24], reducers=[2, 2, 2, 2], **kwargs)


def pointMLPElite(num_classes=40, **kwargs) -> Model:
    return Model(points=1024, class_num=num_classes, embed_dim=32, groups=1, res_expansion=0.25,
                 activation="relu", bias=False, use_xyz=False, normalize="anchor",
                 dim_expansion=[2, 2, 2, 1], pre_blocks=[1, 1, 2, 1], pos_blocks=[1, 1, 2, 1],
                 k_neighbors=[24, 24, 24, 24], reducers=[2, 2, 2, 2], **kwargs)


MODELS = {"pointMLP": pointMLP, "pointMLPElite": pointMLPElite}


def build_model(name: str, num_classes: int = 40, **kwargs) -> Model:
    if name not in MODELS:
        raise KeyError(f"unknown model '{name}'. Valid: {sorted(MODELS)}")
    return MODELS[name](num_classes=num_classes, **kwargs)
