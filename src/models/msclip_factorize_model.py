import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from typing import Any, Dict, List

sys.path.append('MS-CLIP')
from msclip.inference.utils import build_model
from src.models.convlstm import ConvLSTM  # kept in case you later use it


class TemporalEncoderSeq(nn.Module):
    """Transformer over temporal dimension: [B, T, D] -> [B, D]."""
    def __init__(self, embed_dim, num_heads=8, num_layers=2, dropout=0.1, pool="mean"):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pool = pool

        # Optional learned temporal CLS token if you prefer "cls" pooling later
        if pool == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):  # x: [B, T, D]
        if self.pool == "cls":
            B = x.size(0)
            cls_tok = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tok, x], dim=1)  # [B, 1+T, D]
        x = self.encoder(x)  # [B, T, D] or [B, 1+T, D]
        if self.pool == "cls":
            return x[:, 0, :]  # [B, D]
        return x.mean(dim=1)   # [B, D]


class MSClipFactorizeModel(nn.Module):
    def __init__(
        self,
        model_name="Llama3-MS-CLIP-Base",
        ckpt_path=None,
        patch_size: int = 10,
        channels=10,
        num_classes=2,
        out_H=25,
        out_W=25,
        temp_enc_type="attention",   # 'attention' or 'convlstm' (attention recommended for CLS path)
        temp_depth=2,
        use_conv_decoder=True,      # kept for future dense head
        freeze_msclip=True,
        model_config: Dict[str, Any] = None,
        **kwargs,
    ):
        super().__init__()
        self.image_size = model_config["img_res"]
        self.patch_size = model_config["patch_size"]
        # 1) Build MS-CLIP (LightningModule with inference_vision)
        msclip_model, preprocess, tokenizer = build_model(
            model_name=model_name, pretrained=True, ckpt_path=ckpt_path, device="cpu", channels=channels
        )
        self.msclip_model = msclip_model            # has inference_vision(...)
        self.image_encoder = msclip_model.image_encoder  # wrapper (only needed if you later want patch tokens)

        # Flags / meta
        self.freeze_msclip   = freeze_msclip
        self.num_classes     = num_classes
        self.out_H, self.out_W = out_H, out_W
        self.temp_enc_type   = temp_enc_type
        self.use_conv_decoder = use_conv_decoder

        if self.freeze_msclip:
            for p in self.msclip_model.parameters():
                p.requires_grad = False

        # 2) Infer embed_dim using CLS features (not patch tokens)
        with torch.no_grad():
            dummy = torch.zeros(1, channels, 224, 224)
            patch_feats = self.image_encoder.get_patch_embeddings(dummy)  # [1, P, D]
            _, num_patches, embed_dim = patch_feats.shape

        self.embed_dim = embed_dim
        self.num_patches = num_patches

        H_patch = self.image_size // self.patch_size
        W_patch = self.image_size // self.patch_size
        assert H_patch * W_patch == num_patches, \
            f"Grid mismatch: got {num_patches} patches but H_patch*W_patch={H_patch*W_patch}"

        self.H_patch = H_patch
        self.W_patch = W_patch

        # 3) Temporal encoder (sequence over T)
        if temp_enc_type == "attention":
            self.temp_enc = TemporalEncoderSeq(embed_dim=self.embed_dim, num_heads=8, num_layers=temp_depth, dropout=0.1, pool="mean")
        elif temp_enc_type == "convlstm":
            # Note: ConvLSTM expects [B, T, D, H, W]; it doesn't make sense for CLS features.
            # You can keep this path for future patch-token mode; for now, raise to avoid silent misuse.
            raise NotImplementedError("ConvLSTM with CLS features is not supported. Use temp_enc_type='attention'.")
        else:
            raise ValueError(f"Unknown temp_enc_type: {temp_enc_type}")

        # 4) Head: per-sequence classification from [B, D]
        self.head = nn.Linear(self.embed_dim, self.num_classes)

        # Keep place-holders for future dense decoder if you later switch to patch tokens
        self.conv_decoder = None

    def forward(self, batch):
        x = batch   # [B, T, H, W, C]   
        B, T, H, W, C = x.shape
        x = x.permute(0, 1, 4, 2, 3)    # -> [B, T, C, H, W]
        x = x.reshape(B*T, C, H, W)     # [B*T, C, H, W]

        if x.shape[1] > 10:
            x = x[:, :10, :, :]

        # Extract patch embeddings from MS-CLIP
        with torch.no_grad() if self.freeze_msclip else torch.enable_grad():
            patch_feats = self.image_encoder.get_patch_embeddings(x)   # [B*T, P, D]

        # Reshape to [B, P, T, D]
        patch_feats = patch_feats.view(B, self.H_patch, self.W_patch, self.embed_dim)

        patch_feats = patch_feats.permute(0, 2, 1, 3)   # [B, P, T, D]

        # Flatten patches into batch dimension for temporal encoder
        patch_feats = patch_feats.reshape(B * self.num_patches, T, self.embed_dim)  # [B*P, T, D]

        # Temporal encoding
        patch_feats = self.temp_enc(patch_feats)   # [B*P, D] (since we average over T inside)

        # Reshape back to patches
        patch_feats = patch_feats.view(B, self.num_patches, self.embed_dim)  # [B, P, D]

        # Reshape back to grid
        patch_feats = patch_feats.view(B, self.H_patch, self.W_patch, self.embed_dim)  # [B, H_p, W_p, D]
        patch_feats = patch_feats.permute(0, 3, 1, 2)   # [B, D, H_p, W_p]

        # Decode segmentation map
        if self.use_conv_decoder:
            out = self.conv_decoder(patch_feats)        # [B, num_classes, H_p, W_p]
        else:
            out = self.head(patch_feats.mean(dim=[2, 3]))  # [B, num_classes]


        return out

