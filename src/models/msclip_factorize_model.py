import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.append('MS-CLIP')
from msclip.inference.utils import build_model

from src.models.convlstm import ConvLSTM  

class TemporalTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads=8, num_layers=2, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: (B, num_patches, T, D)
        B, P, T, D = x.shape
        x = x.reshape(B * P, T, D)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return x.reshape(B, P, D)

class MSClipFactorizeModel(nn.Module):
    def __init__(
        self,
        model_name="Llama3-MS-CLIP-Base",
        ckpt_path=None,
        channels=14,
        num_classes=2,
        out_H=25,
        out_W=25,
        temp_enc_type="attention",  # 'attention' or 'convlstm'
        temp_depth=2,
        use_conv_decoder=False,
        freeze_msclip=True,
        **kwargs,
    ):
        super().__init__()

        # Build ms-clip model using your existing util function
        msclip_model, preprocess, tokenizer = build_model(
            model_name=model_name, pretrained=True, ckpt_path=ckpt_path, device="cpu", channels=channels
        )
        self.msclip_model = msclip_model  
        self.image_encoder = self.msclip_model.image_encoder  

        # Save some metadata
        self.num_classes = num_classes
        self.out_H = out_H
        self.out_W = out_W
        self.temp_enc_type = temp_enc_type
        self.use_conv_decoder = use_conv_decoder

        # Freeze MS-CLIP weights if requested
        if freeze_msclip:
            for p in self.msclip_model.parameters():
                p.requires_grad = False

        # Determine embed_dim by making a dummy call (safe: run on CPU)
        # Create a tiny dummy input with shape [1, channels, 224, 224]
        with torch.no_grad():
            dummy = torch.zeros(1, channels, 252, 252)

            try:
                patch_feats = self.image_encoder.get_patch_embeddings(dummy)
            except Exception:
                # fallback: attempt using encode_image then error
                raise RuntimeError(
                    "Could not extract patch embeddings during init. Ensure ImageEncoder.get_patch_embeddings works."
                )
            _, num_patches, embed_dim = patch_feats.shape

        self.embed_dim = embed_dim
        self.num_patches = num_patches
        # infer patch grid assuming square
        self.H_patch = int(math.sqrt(num_patches))
        self.W_patch = self.H_patch

        # Temporal encoder
        if temp_enc_type == "convlstm":
            self.temp_enc = ConvLSTM(
                input_dim=self.embed_dim,
                hidden_dim=self.embed_dim,
                kernel_size=(3, 3),
                num_layers=1,
                batch_first=True,
                bias=True,
                return_all_layers=False,
            )
        elif temp_enc_type == "attention":
            self.temp_enc = TemporalTransformer(embed_dim=self.embed_dim, num_heads=8, num_layers=temp_depth, dropout=0.1)
        else:
            raise ValueError(f"Unknown temp_enc_type: {temp_enc_type}")

        # Heads
        if use_conv_decoder:
            # conv decoder: input is (B, embed_dim, H_p, W_p)
            self.conv_decoder = nn.Sequential(
                nn.Conv2d(self.embed_dim, self.embed_dim // 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.embed_dim // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.embed_dim // 2, self.num_classes, kernel_size=1),
            )
            self.head = None
        else:
            # Linear head per patch-token -> later reshape into (B, H_p, W_p, num_classes) -> permute
            self.head = nn.Linear(self.embed_dim, self.num_classes)
            self.conv_decoder = None

    def forward(self, x):
        """
        x: (B, T, C, H, W)
        returns logits: (B, num_classes, out_H, out_W)
        """
        B, T, C, H, W = x.shape
        # collapse batch/time to call MS-CLIP encoder efficiently
        x_in = x.reshape(B * T, C, H, W)  # (B*T, C, H, W)

        # get per-patch embeddings
        # ImageEncoder.get_patch_embeddings expects (B, C, H, W)
        patch_feats = self.image_encoder.get_patch_embeddings(x_in)  # (B*T, P, D)
        P = patch_feats.shape[1]

        # reshape to (B, T, P, D)
        patch_feats = patch_feats.reshape(B, T, P, self.embed_dim)

        if self.temp_enc_type == "convlstm":
            # reshape to (B, T, D, H_p, W_p)
            H_p = self.H_patch
            W_p = self.W_patch
            x_conv = patch_feats.reshape(B, T, H_p, W_p, self.embed_dim).permute(0, 1, 4, 2, 3)
            _, layer_out = self.temp_enc(x_conv)  # returns list of layer outputs, etc.
            # keep last layer's last state
            x_last = layer_out[0][0]  # shape (B, D, H_p, W_p)
            x_feat = x_last  # (B, D, H_p, W_p)
        else:
            # attention temporal encoder: patch_feats -> (B, P, T, D) -> (B, P, D)
            x_att = patch_feats.permute(0, 2, 1, 3)  # (B, P, T, D)
            x_out = self.temp_enc(x_att)  # (B, P, D)
            # reshape to spatial grid
            H_p = self.H_patch
            W_p = self.W_patch
            x_feat = x_out.reshape(B, H_p, W_p, self.embed_dim).permute(0, 3, 1, 2)  # (B, D, H_p, W_p)

        # Head
        if self.use_conv_decoder:
            logits = self.conv_decoder(x_feat)  # (B, num_classes, H_p, W_p)
        else:
            # linear head per token
            # convert to (B, H_p, W_p, D)
            x_l = x_feat.permute(0, 2, 3, 1)  # (B, H_p, W_p, D)
            logits = self.head(x_l)  # (B, H_p, W_p, num_classes)
            logits = logits.permute(0, 3, 1, 2)  # (B, num_classes, H_p, W_p)

        # upsample to requested out_H/out_W
        logits = F.interpolate(logits, size=(self.out_H, self.out_W), mode="bilinear", align_corners=False)
        return logits
