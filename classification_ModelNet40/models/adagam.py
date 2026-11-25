"""AdaGAM: Adaptive Geometric Affine Modulation.

A self-conditioned, zero-initialised residual FiLM for PointMLP's geometric affine.

PointMLP (static GAM), for neighbour features f_{j,k} of anchor a_j in sample b:
    x_{j,k} = (f_{j,k} - a_j) / (sigma_b + eps)        sigma_b = ONE scalar per sample
    out     = alpha * x + beta                          alpha, beta in R^D, fixed at test time

AdaGAM:
    g_j            = [mean_K(x_j) ; max_K(x_j) (; min_K(x_j)) (; std_K(x_j))]   permutation invariant
    [da_j, db_j]   = (bound)( W2 . LeakyReLU_0.2( W1 . g_j ) )                  W2, b2 = 0 at init
    out            = alpha * (1 + da_j) * x + (beta + db_j)

Because fc2 is zero-initialised, an untrained AdaGAM gives outputs that are bit-identical to
static PointMLP. `adapt=False` (or `disable_adaptation=True`) zeroes (da, db) on the same weights:
a counterfactual switch used by the demo and by ablations.

Tensor layout follows PointMLP's grouped tensor: [B, S, K, D] (channels last).
"""
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


def pool_context(x: torch.Tensor, use_std: bool = False, use_min: bool = False) -> torch.Tensor:
    """Symmetric pooling over the neighbour axis K. x: [B, S, K, D] -> [B, S, m*D]."""
    feats = [x.mean(dim=2), x.amax(dim=2)]
    if use_min:
        feats.append(x.amin(dim=2))
    if use_std:
        feats.append(x.var(dim=2, unbiased=False).add(1e-5).sqrt())
    return torch.cat(feats, dim=-1)


class AdaGAM(nn.Module):
    """Predicts per-neighbourhood (d_alpha, d_beta) from the neighbourhood's own statistics.

    Args:
        channels:   feature dimension D of the grouped tensor.
        reduction:  bottleneck ratio r; hidden width = max(16, D // r). Default 8 (+1.35% params on pointMLP).
        use_std:    add std pooling to the context.
        use_min:    add min pooling to the context.
        bound:      squash outputs with tanh (keeps tanh(0) = 0, so identity at init is preserved).
        max_scale:  bound on |d_alpha| when bound=True.
        max_shift:  bound on |d_beta| when bound=True.
        record_stats: store the last call's statistics on the module (not thread-safe; prefer
                      the `aux` dict returned by forward for anything concurrent).
    """

    def __init__(self, channels: int, reduction: int = 8, use_std: bool = False, use_min: bool = False,
                 bound: bool = False, max_scale: float = 1.0, max_shift: float = 2.0,
                 record_stats: bool = False):
        super().__init__()
        if int(channels) <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if int(reduction) <= 0:
            raise ValueError(f"reduction must be positive, got {reduction}")
        if bound and (max_scale <= 0 or max_shift <= 0):
            raise ValueError("max_scale and max_shift must be positive when bound=True")
        self.channels = int(channels)
        self.reduction = int(reduction)
        self.use_std, self.use_min = bool(use_std), bool(use_min)
        self.bound, self.max_scale, self.max_shift = bool(bound), float(max_scale), float(max_shift)
        self.record_stats = bool(record_stats)
        self.disable_adaptation = False

        m = 2 + int(self.use_std) + int(self.use_min)
        self.hidden = max(16, self.channels // self.reduction)
        self.fc1 = nn.Linear(m * self.channels, self.hidden)
        self.act = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(self.hidden, 2 * self.channels)
        nn.init.zeros_(self.fc2.weight)   # identity at init
        nn.init.zeros_(self.fc2.bias)

        self.last_stats: Optional[Dict[str, float]] = None
        self.last_dalpha_per_group: Optional[torch.Tensor] = None
        self.last_context: Optional[torch.Tensor] = None

    def context(self, x: torch.Tensor) -> torch.Tensor:
        return pool_context(x, self.use_std, self.use_min)

    def forward(self, x: torch.Tensor, adapt: bool = True
                ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """x: [B, S, K, D] normalised neighbour features.

        Returns d_alpha, d_beta of shape [B, S, 1, D] and an aux dict (detached tensors):
            context [B, S, m*D], hidden [B, S, H], dalpha_group [B, S], dbeta_group [B, S].
        """
        if x.dim() != 4 or x.shape[-1] != self.channels:
            raise ValueError(f"AdaGAM expects [B, S, K, {self.channels}], got {tuple(x.shape)}")
        B, S, K, D = x.shape
        g = self.context(x)                                   # [B, S, mD]
        h = self.act(self.fc1(g))                             # [B, S, H]
        if adapt and not self.disable_adaptation:
            out = self.fc2(h)                                 # [B, S, 2D]
            d_alpha, d_beta = out[..., :D], out[..., D:]
            if self.bound:
                d_alpha = self.max_scale * torch.tanh(d_alpha / self.max_scale)
                d_beta = self.max_shift * torch.tanh(d_beta / self.max_shift)
        else:
            d_alpha = x.new_zeros(B, S, D)
            d_beta = x.new_zeros(B, S, D)

        aux = {
            "context": g.detach(),
            "hidden": h.detach(),
            "dalpha_group": d_alpha.detach().abs().mean(-1),   # [B, S]
            "dbeta_group": d_beta.detach().abs().mean(-1),     # [B, S]
        }
        if self.record_stats:
            self.last_context = aux["context"]
            self.last_dalpha_per_group = aux["dalpha_group"]
            self.last_stats = {"abs_dalpha": aux["dalpha_group"].mean().item(),
                               "abs_dbeta": aux["dbeta_group"].mean().item()}
        return d_alpha.unsqueeze(2), d_beta.unsqueeze(2), aux

    def extra_repr(self) -> str:
        return (f"channels={self.channels}, reduction={self.reduction}, hidden={self.hidden}, "
                f"use_std={self.use_std}, use_min={self.use_min}, bound={self.bound}")
