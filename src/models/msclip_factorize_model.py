import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from typing import Any, Dict, List, Tuple

sys.path.append('MS-CLIP')
from msclip.inference.utils import build_model
from src.models.convlstm import ConvLSTM  # kept in case you later use it


class TemporalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [B*P, T, D]
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(attn_out + x)   # residual + norm
        return x.mean(dim=1)          # average across time


class LocalScopeSelfAttention(nn.Module):
    """
    Local (k x k) masked self-attention over the spatial patch grid, per frame.
    Input:  x [B, T, H, W, D]
    Output: y [B, T, H, W, D]
    """
    def __init__(self, embed_dim, num_heads=4, k=3, dropout=0.1):
        super().__init__()
        assert k % 2 == 1, "k must be odd"
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.D = embed_dim
        self.k = k
        self.num_heads = num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self._idx_cache: Dict[Tuple[int,int,int], torch.Tensor] = {}  # (H,W,device_idx)->indices

    @torch.no_grad()
    def _build_indices(self, H, W, device):
        # For each (h,w), gather indices of its kxk neighborhood
        r = self.k // 2
        coords = torch.stack(torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device), indexing='ij'), dim=-1)  # [H,W,2]
        nbrs = []
        for dh in range(-r, r+1):
            for dw in range(-r, r+1):
                nh = (coords[...,0] + dh).clamp(0, H-1)
                nw = (coords[...,1] + dw).clamp(0, W-1)
                nbrs.append(nh * W + nw)
        idx = torch.stack(nbrs, dim=0)  # [K, H, W]
        return idx.view(self.k*self.k, H*W)  # [K, N]

    def _get_indices(self, H, W, device):
        key = (H, W, device.index if device.type == "cuda" else -1)
        if key not in self._idx_cache:
            self._idx_cache[key] = self._build_indices(H, W, device)
        return self._idx_cache[key]

    def forward(self, x):
        # x: [B, T, H, W, D]
        B, T, H, W, D = x.shape
        N = H * W
        x_ = x.view(B*T, N, D)  # [BT, N, D]
        x_norm = self.norm(x_)

        Q = self.q_proj(x_norm)   # [BT, N, D]
        K = self.k_proj(x_norm)
        V = self.v_proj(x_norm)

        head_dim = D // self.num_heads
        def split_heads(t):
            return t.view(B*T, N, self.num_heads, head_dim).transpose(1, 2)
        Qh, Kh, Vh = map(split_heads, (Q, K, V))  # each [BT, Hh, N, Dh]

        # --- Build neighbor indices ---
        # self._get_indices returns [K, N] (neighbors per center token, flattened)
        # We want [N, K] to index per-center token the list of neighbors.
        idx = self._get_indices(H, W, x.device).permute(1, 0)  # [N, K]
        Kwin = idx.shape[1]  # = k*k

        # Expand to batch/heads and to match gather's "same-shape" requirement
        # We'll create a dummy K-axis on the SOURCE, then gather along dim=2 (token axis).
        # Source for gather: Kh_exp with shape [BT, Hh, N, K, Dh]
        Kh_exp = Kh.unsqueeze(3).expand(B*T, self.num_heads, N, Kwin, head_dim)
        Vh_exp = Vh.unsqueeze(3).expand(B*T, self.num_heads, N, Kwin, head_dim)

        # Indices: [BT, Hh, N, K, Dh] (same shape as source), gathering along dim=2
        idx_exp = (
            idx.unsqueeze(0).unsqueeze(0)                   # [1,1,N,K]
            .expand(B*T, self.num_heads, N, Kwin)        # [BT,Hh,N,K]
            .unsqueeze(-1)                               # [BT,Hh,N,K,1]
            .expand(B*T, self.num_heads, N, Kwin, head_dim)  # [BT,Hh,N,K,Dh]
        )

        # Gather neighbors along the token axis (dim=2)
        Kh_local = torch.gather(Kh_exp, 2, idx_exp)  # [BT, Hh, N, K, Dh]
        Vh_local = torch.gather(Vh_exp, 2, idx_exp)  # [BT, Hh, N, K, Dh]

        # Attention: Q against local K/V
        # Qh:       [BT, Hh, N, Dh]
        # Kh_local: [BT, Hh, N, K, Dh] -> transpose last two dims to [..., Dh, K]
        scores = torch.matmul(Qh.unsqueeze(3), Kh_local.transpose(-1, -2)).squeeze(3)  # [BT, Hh, N, K]
        scores = scores / (head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        # Weighted sum over K neighbors -> [BT, Hh, N, Dh]
        out = torch.matmul(attn.unsqueeze(3), Vh_local).squeeze(3)  # [BT, Hh, N, Dh]
        out = out.transpose(1, 2).contiguous().view(B*T, N, D)      # [BT, N, D]
        out = self.proj_drop(self.out_proj(out))
        out = out + x_  # residual

        return out.view(B, T, H, W, D)



class MSClipFactorizeModel(nn.Module):
    def __init__(
        self,
        model_name="Llama3-MS-CLIP-Base",
        ckpt_path=None,
        patch_size: int = 16,
        channels=10,
        num_classes=2,
        out_H=25,
        out_W=25,
        temp_enc_type="attention",
        temp_depth=2,
        use_conv_decoder=True,
        freeze_msclip=True,
        use_doy=True,
        use_cls_fusion=False,
        # ---- NEW: LSSA config ----
        use_lssa: bool = False,
        lssa_k: int = 3,
        lssa_heads: int = 4,
        lssa_dropout: float = 0.1,
        # ---------------------------
        image_size: int = 224,
        model_config: Dict[str, Any] = None,
        **kwargs,
    ):
        super().__init__()
        
        self.channels = channels
        self.use_doy = use_doy
        self.image_size = image_size
        self.patch_size = patch_size
        self.use_cls_fusion = use_cls_fusion
        self.use_lssa = use_lssa

        # 1) Build MS-CLIP
        msclip_model, preprocess, tokenizer = build_model(
            model_name=model_name, pretrained=True, ckpt_path=ckpt_path, device="cpu", channels=channels
        )
        self.msclip_model   = msclip_model            
        self.image_encoder  = msclip_model.image_encoder  

        self.freeze_msclip   = freeze_msclip
        self.num_classes     = num_classes
        self.out_H, self.out_W = out_H, out_W
        self.temp_enc_type   = temp_enc_type
        self.use_conv_decoder = use_conv_decoder

        if self.freeze_msclip:
            for p in self.msclip_model.parameters():
                p.requires_grad = False

        # 2) Infer embed_dim and patch grid
        with torch.no_grad():
            dummy = torch.zeros(1, channels, 224, 224)
            patch_feats = self.image_encoder.get_patch_embeddings(dummy)  # [1, P, D]
            _, num_patches, embed_dim = patch_feats.shape

            self.H_patch = self.image_size // self.patch_size
            self.W_patch = self.image_size // self.patch_size

            if num_patches == (self.H_patch * self.W_patch + 1):
                self.has_cls_token = True
                self.num_patches = num_patches - 1
            else:
                self.has_cls_token = False
                self.num_patches = num_patches

            self.embed_dim = embed_dim

        if self.use_doy:
            self.doy_proj = nn.Conv2d(1, self.embed_dim, kernel_size=1)

        # 3) Optional spatial LSSA
        if self.use_lssa:
            self.lssa = LocalScopeSelfAttention(
                embed_dim=self.embed_dim, num_heads=lssa_heads, k=lssa_k, dropout=lssa_dropout
            )

        # 4) Temporal encoder for patch tokens
        if temp_enc_type == "attention":
            self.temp_enc = TemporalAttention(embed_dim=self.embed_dim, num_heads=8, dropout=0.1)
        else:
            raise ValueError(f"Unknown temp_enc_type: {temp_enc_type}")

        # 5) Optional CLS temporal encoder
        if self.use_cls_fusion and self.has_cls_token:
            self.cls_temp_enc = TemporalAttention(embed_dim=self.embed_dim, num_heads=4, dropout=0.1)
            self.cls_fuse_proj = nn.Linear(self.embed_dim, self.embed_dim)

        # 6) Segmentation head
        self.head = nn.Conv2d(self.embed_dim, num_classes, 1)

    def forward(self, batch, doy=None):
        # batch: [B, T, H, W, C]
        B, T, H, W, C = batch.shape
        x = batch.permute(0, 1, 4, 2, 3).reshape(B*T, C, H, W)

        if x.shape[1] > self.channels:
            x = x[:, :self.channels, :, :]

        # === MS-CLIP patch + optional CLS embeddings ===
        with torch.no_grad() if self.freeze_msclip else torch.enable_grad():
            feats = self.image_encoder.get_patch_embeddings(x)  # [B*T, P(+1), D]

        if self.has_cls_token:
            cls_feats = feats[:, 0, :]    # [B*T, D]
            patch_feats = feats[:, 1:, :] # [B*T, P, D]
        else:
            cls_feats, patch_feats = None, feats

        # Reshape to [B, T, P, D]
        patch_feats = patch_feats.view(B, T, self.num_patches, self.embed_dim)

        # === Inject DOY ===
        if self.use_doy and doy is not None:
            doy = doy.float().to(patch_feats.device)
            if doy.ndim == 5:
                doy = doy.permute(0, 1, 4, 2, 3).reshape(B*T, 1, H, W)
            doy = F.interpolate(doy, size=(self.H_patch, self.W_patch), mode="bilinear", align_corners=False)
            doy_feat = self.doy_proj(doy) 
            doy_feat = doy_feat.permute(0, 2, 3, 1).reshape(B*T, self.num_patches, self.embed_dim)
            patch_feats = patch_feats + doy_feat.view(B, T, self.num_patches, self.embed_dim)

        # === Optional Local Scope Self-Attention (spatial, per-frame) ===
        if self.use_lssa:
            # [B,T,P,D] -> [B,T,H_p,W_p,D] -> LSSA -> [B,T,P,D]
            patch_feats = patch_feats.view(B, T, self.H_patch, self.W_patch, self.embed_dim)
            patch_feats = self.lssa(patch_feats)  # spatial cleanup
            patch_feats = patch_feats.view(B, T, self.num_patches, self.embed_dim)

        # === Temporal encoder over T (per-patch location) ===
        patch_feats = patch_feats.permute(0, 2, 1, 3).contiguous()         # [B,P,T,D]
        patch_feats = patch_feats.view(B * self.num_patches, T, self.embed_dim)
        patch_feats = self.temp_enc(patch_feats)                           # [B*P, D]
        patch_feats = patch_feats.view(B, self.num_patches, self.embed_dim)

        # === CLS temporal encoder + fusion (optional) ===
        if self.use_cls_fusion and self.has_cls_token:
            cls_feats = cls_feats.view(B, T, self.embed_dim)               # [B, T, D]
            cls_feats = self.cls_temp_enc(cls_feats)                       # [B, D]
            cls_feats = self.cls_fuse_proj(cls_feats).unsqueeze(1)         # [B, 1, D]
            patch_feats = patch_feats + cls_feats                          # broadcast fusion

        # === Grid reshape and head ===
        patch_feats = patch_feats.view(B, self.H_patch, self.W_patch, self.embed_dim)
        patch_feats = patch_feats.permute(0, 3, 1, 2).contiguous()         # [B,D,H_p,W_p]

        out = self.head(patch_feats)                                       # [B,num_classes,H_p,W_p]
        out = F.interpolate(out, size=(self.out_H, self.out_W), mode="bilinear", align_corners=False)
        return out
