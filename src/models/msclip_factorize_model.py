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
    Swin-style 2D windowed self-attention (no shift, no MLP), per frame.
    Operates on the patch grid, within fixed windows of size w x w.

    Input : x [B, T, H, W, D]  (patch grid per frame)
    Output: y [B, T, H, W, D]  (same shape)

    Notes:
    - If H or W is not divisible by window_size, we pad (like Swin) then unpad.
    - For interpretability, relative position bias is optional (off by default).
    - This is intentionally minimal: LN -> QKV -> MHSA (within windows) -> proj + residual.
    """

    def __init__(self, embed_dim, num_heads=4, window_size=7, dropout=0.0, use_rel_pos_bias=False):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5

        self.norm = nn.LayerNorm(embed_dim)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

        self.use_rel_pos_bias = use_rel_pos_bias
        if use_rel_pos_bias:
            Wh = Ww = window_size
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * Wh - 1) * (2 * Ww - 1), num_heads)
            )
            # precompute pairwise relative position index
            coords_h = torch.arange(Wh)
            coords_w = torch.arange(Ww)
            coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # [2,Wh,Ww]
            coords_flat = torch.flatten(coords, 1)  # [2, Wh*Ww]
            rel = coords_flat[:, :, None] - coords_flat[:, None, :]  # [2, Wh*Ww, Wh*Ww]
            rel = rel.permute(1, 2, 0).contiguous()                   # [Wh*Ww, Wh*Ww, 2]
            rel[:, :, 0] += Wh - 1
            rel[:, :, 1] += Ww - 1
            rel[:, :, 0] *= (2 * Ww - 1)
            relative_position_index = rel.sum(-1)                     # [Wh*Ww, Wh*Ww]
            self.register_buffer("relative_position_index", relative_position_index)
            nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    @staticmethod
    def window_partition(x, window_size):
        # x: [B, H, W, D] -> [B*nW, w, w, D]
        B, H, W, D = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, D)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, D)
        return windows

    @staticmethod
    def window_reverse(windows, window_size, H, W):
        # windows: [B*nW, w, w, D] -> [B, H, W, D]
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def _attend_windows(self, xw):
        """
        xw: [B*nW, w*w, D]
        returns: [B*nW, w*w, D]
        """
        BnW, N, C = xw.shape  # N = w*w
        qkv = self.qkv(xw).reshape(BnW, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # [BnW, nH, N, Dh]

        q = q * self.scale
        attn = torch.matmul(q, k.transpose(-2, -1))  # [BnW, nH, N, N]

        if self.use_rel_pos_bias:
            # add relative biases per head
            Wh = Ww = self.window_size
            rel_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
            rel_bias = rel_bias.view(Wh*Ww, Wh*Ww, -1).permute(2, 0, 1)  # [nH, N, N]
            attn = attn + rel_bias.unsqueeze(0)  # broadcast over batch

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        xw = torch.matmul(attn, v)                       # [BnW, nH, N, Dh]
        xw = xw.transpose(1, 2).reshape(BnW, N, C)       # [BnW, N, D]
        xw = self.proj(xw)
        xw = self.proj_drop(xw)
        return xw

    def forward(self, x):
        # x: [B, T, H, W, D]
        B, T, H, W, D = x.shape
        w = self.window_size

        # process each frame independently for clarity / interpretability
        x = x.view(B*T, H, W, D)
        x = self.norm(x)  # LayerNorm over channels at each patch

        # pad to multiples of window size (like Swin)
        pad_r = (w - W % w) % w
        pad_b = (w - H % w) % w
        if pad_r or pad_b:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))  # pad W then H

        Hp, Wp = x.shape[1], x.shape[2]  # padded sizes

        # partition windows
        xw = self.window_partition(x, w)                # [B*nW, w, w, D]
        xw = xw.view(-1, w*w, D)                        # [B*nW, N, D]

        # attention within windows
        xw = self._attend_windows(xw)                   # [B*nW, N, D]

        # merge windows
        xw = xw.view(-1, w, w, D)                       # [B*nW, w, w, D]
        x = self.window_reverse(xw, w, Hp, Wp)          # [B*T, Hp, Wp, D]

        # unpad
        if pad_r or pad_b:
            x = x[:, :H, :W, :].contiguous()

        # residual (pre-norm)
        # reshape back to [B, T, H, W, D]
        out = x.view(B, T, H, W, D)
        return out




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
        lssa_window = 7,
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

        if self.use_lssa:
            self.lssa = LocalScopeSelfAttention(
                embed_dim=self.embed_dim,
                num_heads=lssa_heads,
                window_size=lssa_window,   
                dropout=lssa_dropout,
                use_rel_pos_bias=False     # set True if you want Swin-like rel bias
            )


        if temp_enc_type == "attention":
            self.temp_enc = TemporalAttention(embed_dim=self.embed_dim, num_heads=8, dropout=0.1)
        else:
            raise ValueError(f"Unknown temp_enc_type: {temp_enc_type}")

        if self.use_cls_fusion and self.has_cls_token:
            self.cls_temp_enc = TemporalAttention(embed_dim=self.embed_dim, num_heads=4, dropout=0.1)
            self.cls_fuse_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.head = nn.Conv2d(self.embed_dim, num_classes, 1)

    def forward(self, batch, doy=None):
        # batch: [B, T, H, W, C]
        B, T, H, W, C = batch.shape
        x = batch.permute(0, 1, 4, 2, 3).reshape(B*T, C, H, W)

        if x.shape[1] > self.channels:
            x = x[:, :self.channels, :, :]

        with torch.no_grad() if self.freeze_msclip else torch.enable_grad():
            feats = self.image_encoder.get_patch_embeddings(x)  # [B*T, P(+1), D]

        if self.has_cls_token:
            cls_feats = feats[:, 0, :]    # [B*T, D]
            patch_feats = feats[:, 1:, :] # [B*T, P, D]
        else:
            cls_feats, patch_feats = None, feats

        # Reshape to [B, T, P, D]
        patch_feats = patch_feats.view(B, T, self.num_patches, self.embed_dim)

        if self.use_doy and doy is not None:
            doy = doy.float().to(patch_feats.device)
            if doy.ndim == 5:
                doy = doy.permute(0, 1, 4, 2, 3).reshape(B*T, 1, H, W)
            doy = F.interpolate(doy, size=(self.H_patch, self.W_patch), mode="bilinear", align_corners=False)
            doy_feat = self.doy_proj(doy) 
            doy_feat = doy_feat.permute(0, 2, 3, 1).reshape(B*T, self.num_patches, self.embed_dim)
            patch_feats = patch_feats + doy_feat.view(B, T, self.num_patches, self.embed_dim)

        if self.use_lssa:
            patch_feats = patch_feats.view(B, T, self.H_patch, self.W_patch, self.embed_dim)
            patch_feats = self.lssa(patch_feats)        
            patch_feats = patch_feats.view(B, T, self.num_patches, self.embed_dim)


        patch_feats = patch_feats.permute(0, 2, 1, 3).contiguous()         # [B,P,T,D]
        patch_feats = patch_feats.view(B * self.num_patches, T, self.embed_dim)
        patch_feats = self.temp_enc(patch_feats)                           # [B*P, D]
        patch_feats = patch_feats.view(B, self.num_patches, self.embed_dim)

        if self.use_cls_fusion and self.has_cls_token:
            cls_feats = cls_feats.view(B, T, self.embed_dim)               # [B, T, D]
            cls_feats = self.cls_temp_enc(cls_feats)                       # [B, D]
            cls_feats = self.cls_fuse_proj(cls_feats).unsqueeze(1)         # [B, 1, D]
            patch_feats = patch_feats + cls_feats                         
        patch_feats = patch_feats.view(B, self.H_patch, self.W_patch, self.embed_dim)
        patch_feats = patch_feats.permute(0, 3, 1, 2).contiguous()         # [B,D,H_p,W_p]

        out = self.head(patch_feats)  # [B, num_classes, H_p, W_p] (H_p = H//num_patches)
        out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)

        return out
