"""
AdaGAM: Adaptive Geometric Affine Modulation.

Drop-in replacement for the static affine step inside PointMLP's LocalGrouper.

PointMLP (static GAM), for grouped neighbours f_{j,k} of anchor a_j in sample b:
    f~_{j,k} = (f_{j,k} - a_j) / (sigma_b + eps)       sigma_b: ONE scalar per sample
    f^_{j,k} = alpha * f~_{j,k} + beta                  alpha, beta in R^D, frozen at test time

AdaGAM (this file):
    g_j      = [AvgPool_K(s_j) ; MaxPool_K(s_j) (; StdPool_K(s_j))]     permutation invariant
    [da, db] = W2 LeakyReLU(W1 g_j)                                      W2 zero-initialised
    f^_{j,k} = alpha * (1 + da_j) * f~_{j,k} + (beta + db_j)

    s_j is the normalised patch f~_j in "pointmlp" mode, the raw patch f_j in "patch" mode.
    Because W2 starts at zero, AdaGAM is *exactly* static GAM at initialisation, so any
    difference after training comes from the adaptive part.

norm="pointmlp": keep PointMLP's anchor / per-sample-std normalisation (clean ablation).
norm="patch":    per-channel, per-patch mean/std over K (the formulation in the MTE handoff).
"""

import torch
import torch.nn as nn


class AdaGAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 4, use_std: bool = False,
                 norm: str = "pointmlp", eps: float = 1e-5):
        super().__init__()
        assert norm in ("pointmlp", "patch"), norm
        self.channels = channels
        self.use_std = use_std
        self.norm = norm
        self.eps = eps

        # static part (identical to PointMLP's affine_alpha / affine_beta)
        self.affine_alpha = nn.Parameter(torch.ones([1, 1, 1, channels]))
        self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, channels]))

        in_dim = channels * (3 if use_std else 2)
        hidden = max(16, channels // reduction)
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.fc2 = nn.Linear(hidden, 2 * channels)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

        # detached statistics of the last forward pass, for analysis plots
        self.record_stats = False
        self.disable_adaptation = False
        self.last_stats = None
        self.last_dalpha_per_group = None
        self.last_context = None

    def _context(self, s: torch.Tensor) -> torch.Tensor:
        # s: [B, S, K, C] -> [B, S, 2C or 3C]
        feats = [s.mean(dim=2), s.amax(dim=2)]
        if self.use_std:
            feats.append(torch.sqrt(s.var(dim=2, unbiased=False) + self.eps))
        return torch.cat(feats, dim=-1)

    def forward(self, grouped: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        """
        grouped: [B, S, K, C] neighbour features
        anchor:  [B, S, 1, C] anchor (centroid) features
        returns: [B, S, K, C]
        """
        B = grouped.shape[0]
        if self.norm == "pointmlp":
            centered = grouped - anchor
            std = torch.std(centered.reshape(B, -1), dim=-1, keepdim=True)[..., None, None]
            x_norm = centered / (std + self.eps)
            ctx_src = x_norm
        else:
            mean = grouped.mean(dim=2, keepdim=True)
            std = torch.sqrt(grouped.var(dim=2, keepdim=True, unbiased=False) + self.eps)
            x_norm = (grouped - mean) / std
            ctx_src = grouped  # normalised patch has mean 0 / std 1, so pool the raw one

        g = self._context(ctx_src)                       # [B, S, in_dim]
        if self.disable_adaptation:
            d_alpha = torch.zeros(B, g.shape[1], 1, self.channels, device=g.device, dtype=g.dtype)
            d_beta = torch.zeros_like(d_alpha)
        else:
            d_alpha, d_beta = self.fc2(self.act(self.fc1(g))).chunk(2, dim=-1)  # [B, S, C] each
            d_alpha = d_alpha.unsqueeze(2)               # [B, S, 1, C]
            d_beta = d_beta.unsqueeze(2)

        if self.record_stats:
            self.last_stats = {
                "d_alpha_abs": d_alpha.detach().abs().mean().item(),
                "d_beta_abs": d_beta.detach().abs().mean().item(),
                "d_alpha_std": d_alpha.detach().std().item(),
            }
            self.last_dalpha_per_group = d_alpha.detach().abs().mean(dim=-1).squeeze(-1)  # [B, S]
            self.last_context = g.detach()  # [B, S, in_dim]

        return self.affine_alpha * (1.0 + d_alpha) * x_norm + (self.affine_beta + d_beta)
