
"""
adapters.py
-------------
Lightweight, interpretable adapters for aligning Sentinel-2 L1C inputs to MS-CLIP L2A feature space.

Variants:
  - Embedding-space (recommended for XAI & simplicity):
      * EmbedAdapterAffine  : per-channel affine (max interpretability)
      * EmbedAdapterDLR     : diagonal + low-rank linear map
      * EmbedAdapterWCT     : BN-style whitening + low-rank coloring
  - Pixel-space (optional; interpretable radiometric calibration before encoder):
      * PixelAdapterTiny    : affine + depthwise 3x3 + pointwise 1x1 (residual)
      * PixelAdapterTinyPP  : Tiny++ with dilations, SE-lite, LayerNorm (residual)

A small factory `build_adapter` is provided.

All adapters implement `.identity_reg()` for near-identity regularization (safe default: small or zero).
"""

from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "EmbedAdapterAffine",
    "EmbedAdapterDLR",
    "EmbedAdapterWCT",
    "BandAffine",
    "SE_Lite",
    "PixelAdapterTiny",
    "PixelAdapterTinyPP",
    "build_adapter",
]


# =============================
# Embedding-space adapters
# =============================

class EmbedAdapterAffine(nn.Module):
    """Per-channel affine in embedding space (max interpretability).

    Forward:  z_hat = z * alpha + beta   (broadcast along tokens)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim))
        self.beta  = nn.Parameter(torch.zeros(dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.alpha + self.beta

    def identity_reg(self) -> torch.Tensor:
        # Encourage near-identity (alpha≈1, beta≈0)
        return ((self.alpha - 1.0) ** 2).mean() + (self.beta ** 2).mean()


class EmbedAdapterDLR(nn.Module):
    """Diagonal + Low-Rank (DLR) linear map in embedding space.

    Forward: z_hat = z * diag + (z @ U) @ V^T + bias
    where U,V have small rank r (default 16).
    """
    def __init__(self, dim: int, rank: int = 16):
        super().__init__()
        self.diag = nn.Parameter(torch.ones(dim))      # starts at identity
        self.U = nn.Parameter(torch.zeros(dim, rank))  # low-rank
        self.V = nn.Parameter(torch.zeros(dim, rank))
        self.bias = nn.Parameter(torch.zeros(dim))

        # small near-zero init for U, V
        nn.init.normal_(self.U, std=1e-3)
        nn.init.normal_(self.V, std=1e-3)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        orig_shape = z.shape  # [..., C]
        zf = z.reshape(-1, orig_shape[-1])             # [N, C]
        zD = zf * self.diag                            # diag part
        zLR = (zf @ self.U) @ self.V.t()               # low-rank part
        out = zD + zLR + self.bias
        return out.view(orig_shape)

    def identity_reg(self) -> torch.Tensor:
        # Keep near identity: diag≈1, U,V≈0, bias≈0
        return ((self.diag - 1.0) ** 2).mean() + (self.U ** 2).mean() + (self.V ** 2).mean() + (self.bias ** 2).mean()


class EmbedAdapterWCT(nn.Module):
    """Whitening–Coloring Transform (lightweight, learned).

    Maintains running source mean/var (BN-style) and target mean.
    Applies:  z_norm = (z - mu_src) / sqrt(var_src + eps)
              z_col  = (I + U V^T) z_norm
              z_hat  = z_col * scale + bias + mu_tgt

    Notes:
      * Call `update_stats(z_src, z_tgt)` during training to track running stats.
      * At eval, forward uses the frozen running statistics.
    """
    def __init__(self, dim: int, rank: int = 16, momentum: float = 0.01, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.momentum = float(momentum)
        self.eps = float(eps)

        # Running stats (buffers, not learnable)
        self.register_buffer("running_mean_src", torch.zeros(dim))
        self.register_buffer("running_var_src",  torch.ones(dim))
        self.register_buffer("running_mean_tgt", torch.zeros(dim))

        # Low-rank coloring (I + U V^T), plus optional scale/bias
        self.U = nn.Parameter(torch.zeros(dim, rank))
        self.V = nn.Parameter(torch.zeros(dim, rank))
        self.scale = nn.Parameter(torch.ones(dim))
        self.bias  = nn.Parameter(torch.zeros(dim))

        nn.init.normal_(self.U, std=1e-3)
        nn.init.normal_(self.V, std=1e-3)

    @torch.no_grad()
    def update_stats(self, z_src: torch.Tensor, z_tgt: torch.Tensor) -> None:
        """Update running source mean/var and target mean from current batch (flattened over tokens)."""
        zsrc = z_src.reshape(-1, z_src.shape[-1])
        ztgt = z_tgt.reshape(-1, z_tgt.shape[-1])

        m_src = zsrc.mean(dim=0)
        v_src = zsrc.var(dim=0, unbiased=False)
        m_tgt = ztgt.mean(dim=0)

        self.running_mean_src.lerp_(m_src, self.momentum)
        self.running_var_src.lerp_(v_src,  self.momentum)
        self.running_mean_tgt.lerp_(m_tgt, self.momentum)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Normalize using running source stats
        z_hat = (z - self.running_mean_src) / torch.sqrt(self.running_var_src + self.eps)
        # Low-rank coloring
        orig = z_hat
        z_hat = z_hat.reshape(-1, self.dim)
        z_hat = z_hat + (z_hat @ self.U) @ self.V.t()   # (I + U V^T) z
        z_hat = z_hat * self.scale + self.bias
        z_hat = z_hat.view_as(orig)
        # Recenter to target mean
        return z_hat + self.running_mean_tgt

    def identity_reg(self) -> torch.Tensor:
        # Keep near identity transform
        return ((self.scale - 1.0) ** 2).mean() + (self.bias ** 2).mean() + (self.U ** 2).mean() + (self.V ** 2).mean()


# =============================
# Pixel-space adapters (optional)
# =============================

class BandAffine(nn.Module):
    """Per-band affine (radiometric) calibration in pixel space."""
    def __init__(self, channels: int):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gain + self.bias


class SE_Lite(nn.Module):
    """Squeeze-and-Excite (lite) for per-scene band reweighting."""
    def __init__(self, channels: int, r: int = 8):
        super().__init__()
        hidden = max(1, channels // r)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.gelu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w


class PixelAdapterTiny(nn.Module):
    """Tiny pixel adapter: Affine → DWConv3×3 → GELU → PWConv1×1 → Residual."""
    def __init__(self, channels: int):
        super().__init__()
        C = channels
        self.aff = BandAffine(C)
        self.dw = nn.Conv2d(C, C, kernel_size=3, padding=1, groups=C, bias=False)
        self.pw = nn.Conv2d(C, C, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.aff(x)
        y = self.dw(x0)
        y = F.gelu(self.pw(y))
        return y + x0

    def identity_reg(self) -> torch.Tensor:
        # Rely on weight decay for near-identity; explicit reg not needed.
        return torch.zeros((), device=next(self.parameters()).device)


class PixelAdapterTinyPP(nn.Module):
    """Tiny++ pixel adapter: Affine → (DW3×3 || DW3×3 dil=3) → concat → 1×1 → GELU → LN → SE-lite → 1×1 → Residual."""
    def __init__(self, channels: int):
        super().__init__()
        C = channels
        self.aff = BandAffine(C)
        self.dw3   = nn.Conv2d(C, C, kernel_size=3, padding=1, groups=C, bias=False)
        self.dw3d  = nn.Conv2d(C, C, kernel_size=3, padding=3, dilation=3, groups=C, bias=False)
        self.mix1  = nn.Conv2d(2 * C, C, kernel_size=1, bias=False)
        self.ln    = nn.LayerNorm(C)  # channels-last usage
        self.se    = SE_Lite(C, r=8)
        self.mix2  = nn.Conv2d(C, C, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.aff(x)
        a = self.dw3(x0)
        b = self.dw3d(x0)
        y = torch.cat([a, b], dim=1)
        y = F.gelu(self.mix1(y))
        # channels-last LayerNorm
        y = y.permute(0, 2, 3, 1)
        y = self.ln(y)
        y = y.permute(0, 3, 1, 2)
        y = self.se(y)
        y = self.mix2(y)
        return y + x0

    def identity_reg(self) -> torch.Tensor:
        return torch.zeros((), device=next(self.parameters()).device)


# =============================
# Factory
# =============================

def build_adapter(
    model_type: str,
    embed_dim: int,
    in_channels: int,
    rank: int = 16,
    wct_momentum: float = 0.01,
) -> Tuple[nn.Module, str]:
    """Factory to build an adapter by name.

    Args:
        model_type: one of
            ["embed_affine","embed_dlr","embed_wct","linear","pixel_tiny","pixel_tinypp"]
        embed_dim: embedding dimension (for embedding-space variants)
        in_channels: number of spectral channels (for pixel-space variants)
        rank: low-rank size for DLR/WCT
        wct_momentum: running-stat momentum for WCT
    Returns:
        (adapter_module, adapter_space) where adapter_space in {"embedding", "pixel"}
    """
    mt = model_type.lower()
    if mt in ("embed_affine", "affine"):
        return EmbedAdapterAffine(embed_dim), "embedding"
    if mt in ("embed_dlr", "dlr"):
        return EmbedAdapterDLR(embed_dim, rank=rank), "embedding"
    if mt in ("embed_wct", "wct"):
        return EmbedAdapterWCT(embed_dim, rank=rank, momentum=wct_momentum), "embedding"
    if mt == "linear":
        # Backwards-compat: simple linear layer as identity-ish map
        layer = nn.Linear(embed_dim, embed_dim, bias=True)
        nn.init.eye_(layer.weight) if hasattr(nn.init, "eye_") and layer.weight.shape[0] == layer.weight.shape[1] else nn.init.kaiming_uniform_(layer.weight, a=5 ** 0.5)
        nn.init.zeros_(layer.bias)
        class LinearWrapper(nn.Module):
            def __init__(self, layer):
                super().__init__()
                self.layer = layer
            def forward(self, z):
                return self.layer(z)
            def identity_reg(self) -> torch.Tensor:
                return (self.layer.weight - torch.eye(self.layer.weight.shape[0], device=self.layer.weight.device)).pow(2).mean() + (self.layer.bias ** 2).mean()
        return LinearWrapper(layer), "embedding"
    if mt == "pixel_tiny":
        return PixelAdapterTiny(in_channels), "pixel"
    if mt == "pixel_tinypp":
        return PixelAdapterTinyPP(in_channels), "pixel"
    raise ValueError(f"Unknown model_type '{model_type}'.")
