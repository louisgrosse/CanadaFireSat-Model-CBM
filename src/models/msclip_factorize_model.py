import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from typing import Any, Dict

from src.models.l1c2l2a_adapter import L1C2L2AAdapter

sys.path.append('MS-CLIP')
from msclip.inference.utils import build_model
from msclip.inference.clearclip import maybe_patch_clearclip


# -----------------------
# DOY Encoder
# -----------------------
class DOYEmbed(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.fc = nn.Linear(2, embed_dim)

    def forward(self, doy):
        # doy: [B, T] integers (1–365)
        theta = 2 * math.pi * doy / 365.0
        sin = torch.sin(theta)
        cos = torch.cos(theta)
        cyc = torch.stack([sin, cos], dim=-1)   # [B, T, 2]
        return self.fc(cyc)                     # [B, T, D]


# -----------------------
# Temporal Attention with DOY conditioning
# -----------------------
class TemporalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, doy_emb=None):
        # x: [B*P, T, D]
        # doy_emb: [B*P, T, D] 
        if doy_emb is not None:
            q_in = x + doy_emb
            k_in = x + doy_emb
        else:
            q_in = k_in = x

        attn_out, _ = self.attn(q_in, k_in, x) 
        x = self.norm(attn_out + x)
        return x.mean(dim=1) 


# -----------------------
# Main Model
# -----------------------
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
        freeze_msclip=True,
        use_doy=True,
        ds_labels=False,
        use_cls_fusion=False,
        image_size: int = 224,
        use_l1c2l2a_adapter: bool = False,
        l1c2l2a_dropout: int = 0,
        l1c2l2a_Adapter_loc:str = "",
        model_config: Dict[str, Any] = None,
        **kwargs,
    ):
        super().__init__()
        
        self.ds_labels = ds_labels
        self.out_H = out_H
        self.out_W = out_W
        self.channels = channels
        self.use_doy = use_doy
        self.image_size = image_size
        self.patch_size = patch_size
        self.use_cls_fusion = use_cls_fusion

        # -----------------------
        # Load MS-CLIP
        # -----------------------
        msclip_model, preprocess, tokenizer = build_model(
            model_name=model_name, pretrained=True, ckpt_path=ckpt_path, device="cpu", channels=channels
        )
        self.msclip_model   = msclip_model            
        self.image_encoder  = msclip_model.image_encoder  

        if model_config is not None and "clearclip" in model_config:
            num_patched = maybe_patch_clearclip(self.image_encoder, model_config["clearclip"])
            if num_patched > 0:
                print(f"[ClearCLIP] Patched last {num_patched} vision blocks "
                    f"(keep_ffn={model_config['clearclip'].get('keep_ffn', False)}, "
                    f"keep_residual={model_config['clearclip'].get('keep_residual', False)})")

        if freeze_msclip:
            for p in self.msclip_model.parameters():
                p.requires_grad = False

        # determine patch dims
        with torch.no_grad():
            dummy = torch.zeros(1, channels, 224, 224)
            patch_feats = self.image_encoder.get_patch_embeddings(dummy)  # [1, P, D]
            _, num_patches, embed_dim = patch_feats.shape
            self.embed_dim = embed_dim
            self.H_patch = self.image_size // self.patch_size
            self.W_patch = self.image_size // self.patch_size

            if num_patches == (self.H_patch * self.W_patch + 1):
                self.has_cls_token = True
                self.num_patches = num_patches - 1
            else:
                self.has_cls_token = False
                self.num_patches = num_patches

        self.use_l1c2l2a_adapter = use_l1c2l2a_adapter
        self.l1c2l2a_dropout = l1c2l2a_dropout

        if self.use_l1c2l2a_adapter:
            self.l1c2l2a = L1C2L2AAdapter(dim=self.embed_dim, dropout=self.l1c2l2a_dropout)
            adapter_weights = torch.load("/home/louis/Code/wildfire-forecast/worldstrat/l1c2l2a_linear.pt", map_location="cpu")
            self.l1c2l2a.load_state_dict(adapter_weights)
        
        # DOY encoder
        if self.use_doy:
            self.doy_embed = DOYEmbed(self.embed_dim)

        # Temporal attention
        if temp_enc_type == "attention":
            self.temp_enc = TemporalAttention(embed_dim=self.embed_dim, num_heads=8, dropout=0.1)
        else:
            raise ValueError(f"Unknown temp_enc_type: {temp_enc_type}")

        # Optional CLS fusion
        if self.use_cls_fusion and self.has_cls_token:
            self.cls_temp_enc = TemporalAttention(embed_dim=self.embed_dim, num_heads=4, dropout=0.1)
            self.cls_fuse_proj = nn.Linear(self.embed_dim, self.embed_dim)

        # Segmentation head
        self.head = nn.Conv2d(self.embed_dim, num_classes, 1)

    def forward(self, batch, doy=None):
        # EXPECTED LAYOUT: [B, T, C, H, W] already normalized (MS-CLIP stats)
        assert batch.ndim == 5, f"inputs must be [B,T,C,H,W], got {batch.ndim} dims"
        B, T, C, H, W = batch.shape

        # Basic checks
        assert C == self.channels, f"channels mismatch: got {C}, expected {self.channels}"
        assert H == self.image_size and W == self.image_size, f"spatial mismatch: ({H},{W}) vs {self.image_size}"
        assert torch.isfinite(batch).all(), "inputs contain NaN/Inf"
        assert batch.dtype in (torch.float16, torch.float32, torch.bfloat16), f"bad dtype {batch.dtype}"

        # Flatten time for the encoder
        x = batch.reshape(B * T, C, H, W)

        # Extra guard: image encoder expects patch_size <= H,W
        assert self.patch_size <= H and self.patch_size <= W, f"patch {self.patch_size} > ({H},{W})"

        with torch.no_grad():
            feats = self.image_encoder.get_patch_embeddings(x)  # [B*T, P(+1), D]
            assert torch.isfinite(feats).all(), "encoder outputs contain NaN/Inf"

        # split CLS vs patches
        if self.has_cls_token:
            cls_feats = feats[:, 0, :]            # [B*T, D]
            patch_feats = feats[:, 1:, :]         # [B*T, P, D]
        else:
            cls_feats, patch_feats = None, feats

        # [B, T, P, D]
        patch_feats = patch_feats.view(B, T, self.num_patches, self.embed_dim)

        # Optional adapter
        if self.use_l1c2l2a_adapter:
            patch_feats = self.l1c2l2a(patch_feats.view(B*T, self.num_patches, self.embed_dim)).view(
                B, T, self.num_patches, self.embed_dim
            )

        # Temporal attention (B*P, T, D)
        patch_feats = patch_feats.permute(0, 2, 1, 3).contiguous().view(B * self.num_patches, T, self.embed_dim)

        # DOY checks
        if self.use_doy and doy is not None:
            if doy.ndim > 2:  # [B,T,H,W,1] -> [B,T]
                doy = doy.view(B, T, -1)[:, :, 0]
            assert doy.shape == (B, T), f"DOY must be [B,T], got {tuple(doy.shape)}"
            assert torch.isfinite(doy).all(), "DOY has NaN/Inf"
            # clamp to valid range just in case (won't hurt)
            doy = doy.clamp(1, 366)
            doy_emb = self.doy_embed(doy)  # [B,T,D]
            doy_emb = doy_emb.unsqueeze(1).expand(-1, self.num_patches, -1, -1).reshape(
                B * self.num_patches, T, self.embed_dim
            )
        else:
            doy_emb = None

        # Temporal encoding
        patch_feats = self.temp_enc(patch_feats, doy_emb=doy_emb)  # [B*P, D]
        assert torch.isfinite(patch_feats).all(), "Temporal encoder produced NaN/Inf"

        # [B, P, D] -> grid
        patch_feats = patch_feats.view(B, self.num_patches, self.embed_dim)
        if self.use_cls_fusion and self.has_cls_token:
            cls_feats = cls_feats.view(B, T, self.embed_dim)
            cls_feats = self.cls_temp_enc(cls_feats)
            patch_feats = patch_feats + self.cls_fuse_proj(cls_feats).unsqueeze(1)

        # [B,D,H_p,W_p]
        patch_feats = patch_feats.view(B, self.H_patch, self.W_patch, self.embed_dim).permute(0, 3, 1, 2).contiguous()

        out = self.head(patch_feats)  # [B, num_classes, H_p, W_p]
        assert torch.isfinite(out).all(), "Head produced NaN/Inf"

        # Up-sample to match labels size outside (caller will compare); default to H,W of input
        if not self.ds_labels:
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        else:
            out = F.interpolate(out, size=(self.out_H, self.out_W), mode="bilinear", align_corners=False)

        return out
