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
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, doy_emb=None):
        # x: [B*P, T, D]
        # doy_emb: [B*P, T, D] (same size, from your sin/cos DOY projection)

        if doy_emb is not None:
            q_in = x + doy_emb
            k_in = x + doy_emb
        else:
            q_in = k_in = x

        attn_out, _ = self.attn(q_in, k_in, x)  # note: V stays as x
        x = self.norm(attn_out + x)
        return x.mean(dim=1)


class DOYEmbed(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.fc = nn.Linear(2, embed_dim)

    def forward(self, doy):
        # doy: [B, T] int
        theta = 2 * math.pi * doy / 365.0
        sin = torch.sin(theta)
        cos = torch.cos(theta)
        cyc = torch.stack([sin, cos], dim=-1)   # [B, T, 2]
        return self.fc(cyc)                     # [B, T, D]


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
