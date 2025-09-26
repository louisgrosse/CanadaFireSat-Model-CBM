import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from typing import Any, Dict, List

sys.path.append('MS-CLIP')
from msclip.inference.utils import build_model
from src.models.convlstm import ConvLSTM  # kept in case you later use it


class TemporalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [B*P, T, D]
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(attn_out + x)   # residual + norm
        return x.mean(dim=1)          # average across time



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
        temp_enc_type="attention",   # 'attention' or 'convlstm' (attention recommended for CLS path)
        temp_depth=2,
        use_conv_decoder=True,      # kept for future dense head
        freeze_msclip=True,
        use_doy=True,
        image_size: int = 224,
        model_config: Dict[str, Any] = None,
        **kwargs,
    ):
        super().__init__()
        
        self.channels = channels
        self.use_doy = use_doy
        self.image_size = image_size
        self.patch_size = patch_size
        # 1) Build MS-CLIP (LightningModule with inference_vision)
        msclip_model, preprocess, tokenizer = build_model(
            model_name=model_name, pretrained=True, ckpt_path=ckpt_path, device="cpu", channels=channels
        )
        self.msclip_model = msclip_model            
        self.image_encoder = msclip_model.image_encoder  # wrapper (only needed if you later want patch tokens)

        self.freeze_msclip   = freeze_msclip
        self.num_classes     = num_classes
        self.out_H, self.out_W = out_H, out_W
        self.temp_enc_type   = temp_enc_type
        self.use_conv_decoder = use_conv_decoder

        if self.freeze_msclip:
            for p in self.msclip_model.parameters():
                p.requires_grad = False

        # 2) Infer embed_dim
        with torch.no_grad():
            dummy = torch.zeros(1, channels, 224, 224)
            patch_feats = self.image_encoder.get_patch_embeddings(dummy)  # [1, P, D]
            _, num_patches, embed_dim = patch_feats.shape

            self.H_patch = self.image_size // self.patch_size
            self.W_patch = self.image_size // self.patch_size

            # Drop CLS (if present)
            if num_patches == (self.H_patch * self.W_patch + 1):
                num_patches_no_cls = num_patches - 1
                self.has_cls_token = True
            else:
                num_patches_no_cls = num_patches
                self.has_cls_token = False

            self.embed_dim = embed_dim
            self.num_patches = num_patches_no_cls

            print(f"[DEBUG] num_patches={num_patches}, num_patches_no_cls={num_patches_no_cls}, "
                f"H_patch={self.H_patch}, W_patch={self.W_patch}, CLS={self.has_cls_token}")

            assert self.H_patch * self.W_patch == self.num_patches, \
                f"Grid mismatch: got {num_patches} tokens ({'with' if self.has_cls_token else 'no'} CLS), " \
                f"but H_patch*W_patch={self.H_patch*self.W_patch}"

        if self.use_doy:
            self.doy_proj = nn.Conv2d(1, self.embed_dim, kernel_size=1)


        # 3) Temporal encoder (sequence over T)
        if temp_enc_type == "attention":
            self.temp_enc = TemporalAttention(embed_dim=self.embed_dim, num_heads=8, dropout=0.1)
        elif temp_enc_type == "convlstm":

            raise NotImplementedError("ConvLSTM with CLS features is not supported. Use temp_enc_type='attention'.")
        else:
            raise ValueError(f"Unknown temp_enc_type: {temp_enc_type}")

        self.head = nn.Conv2d(self.embed_dim, num_classes, 1)

    def forward(self, batch, doy=None):
        # Input: [B, T, H, W, C]
        x = batch
        B, T, H, W, C = x.shape

        # Rearrange to [B*T, C, H, W]
        x = x.permute(0, 1, 4, 2, 3).reshape(B*T, C, H, W)

        # Just in case Doy was added in the batch before it should have.
        if x.shape[1] > self.channels:
            x = x[:, :self.channels, :, :]

        # === MS-CLIP patch embeddings ===
        with torch.no_grad() if self.freeze_msclip else torch.enable_grad():
            patch_feats = self.image_encoder.get_patch_embeddings(x)  # [B*T, P, D]

        # Drop CLS token if present
        if getattr(self, "has_cls_token", True):
            patch_feats = patch_feats[:, 1:, :]   # [B*T, P, D]

        # Reshape to [B, T, P, D]
        patch_feats = patch_feats.view(B, T, self.num_patches, self.embed_dim)

        # === Inject DOY here (before temporal encoder) ===
        if self.use_doy and doy is not None:
            doy = doy.float().to(patch_feats.device)  # [B, T, H, W, 1]

            if doy.ndim == 5:  # [B, T, H, W, 1]
                # [B, T, H, W, 1] -> [B*T, 1, H, W]
                doy = doy.permute(0, 1, 4, 2, 3).reshape(B*T, 1, H, W)

            # Downsample to patch grid size [B*T, 1, H_p, W_p]
            doy = F.interpolate(doy, size=(self.H_patch, self.W_patch), mode="bilinear", align_corners=False)

            # Project to embedding dimension
            doy_feat = self.doy_proj(doy)  # [B*T, D, H_p, W_p]

            # Reshape doy_feat to patch-token form: [B*T, P, D]
            doy_feat = doy_feat.permute(0, 2, 3, 1).reshape(B*T, self.num_patches, self.embed_dim)

            # Add to patch embeddings
            patch_feats = patch_feats + doy_feat.view(B, T, self.num_patches, self.embed_dim)

        # === Temporal encoder ===
        # Rearrange for temporal encoder: [B, P, T, D]
        patch_feats = patch_feats.permute(0, 2, 1, 3).contiguous()
        # Flatten patches into batch: [B*P, T, D]
        patch_feats = patch_feats.view(B * self.num_patches, T, self.embed_dim)
        patch_feats = self.temp_enc(patch_feats)   # [B*P, D]

        # Reshape back: [B, P, D]
        patch_feats = patch_feats.view(B, self.num_patches, self.embed_dim)

        # Reshape to grid [B, H_p, W_p, D] → [B, D, H_p, W_p]
        patch_feats = patch_feats.view(B, self.H_patch, self.W_patch, self.embed_dim)
        patch_feats = patch_feats.permute(0, 3, 1, 2).contiguous()

        # === Segmentation head ===
        out = self.head(patch_feats)  # [B, num_classes, H_p, W_p]

        # Upsample to ground-truth resolution
        out = F.interpolate(out, size=(self.out_H, self.out_W), mode="bilinear", align_corners=False)

        return out

