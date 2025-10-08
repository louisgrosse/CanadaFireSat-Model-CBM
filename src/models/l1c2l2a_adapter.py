import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from typing import Dict, Any

sys.path.append("MS-CLIP")
from msclip.inference.utils import build_model
from msclip.inference.clearclip import maybe_patch_clearclip


# -----------------------
# Adapter Layer
# -----------------------
class L1C2L2AAdapter(nn.Module):
    """Linear transform in CLIP embedding space (D×D)."""
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B,P,D] or [B*P,D]
        x = self.linear(x)
        x = self.dropout(x)
        return x


# -----------------------
# Main Model
# -----------------------
class L1C2L2AAdapterModel(nn.Module):
    """
    Minimal training model to learn the linear mapping that converts
    L1C embeddings into L2A embeddings in MS-CLIP feature space.
    Compatible with the segmentation training pipeline.
    """

    def __init__(
        self,
        model_name="Llama3-MS-CLIP-Base",
        ckpt_path=None,
        freeze_msclip=True,
        channels=10,
        image_size=224,
        model_config: Dict[str, Any] = None,
        **kwargs,
    ):
        super().__init__()

        # -----------------------
        # Load pretrained MS-CLIP encoder
        # -----------------------
        msclip_model, preprocess, tokenizer = build_model(
            model_name=model_name,
            pretrained=True,
            ckpt_path=ckpt_path,
            device="cpu",
            channels=channels,
        )
        self.msclip_model = msclip_model
        self.image_encoder = msclip_model.image_encoder

        if model_config is not None and "clearclip" in model_config:
            num_patched = maybe_patch_clearclip(self.image_encoder, model_config["clearclip"])
            if num_patched > 0:
                print(
                    f"[ClearCLIP] Patched last {num_patched} vision blocks "
                    f"(keep_ffn={model_config['clearclip'].get('keep_ffn', False)}, "
                    f"keep_residual={model_config['clearclip'].get('keep_residual', False)})"
                )

        if freeze_msclip:
            for p in self.msclip_model.parameters():
                p.requires_grad = False

        # Determine embedding dimension
        with torch.no_grad():
            dummy = torch.zeros(1, channels, image_size, image_size)
            patch_feats = self.image_encoder.get_patch_embeddings(dummy)  # [1,P(+1),D]
            _, num_patches, embed_dim = patch_feats.shape
        self.embed_dim = embed_dim

        # Define the adapter
        dropout = model_config.get("l1c2l2a_dropout", 0.1) if model_config else 0.1
        self.adapter = L1C2L2AAdapter(dim=self.embed_dim, dropout=dropout)

    # -----------------------
    # Forward (two inputs: L1C & L2A)
    # -----------------------
    def forward(self, batch):
        """
        batch: dict with
            batch["l1c"]: [B,C,H,W]
            batch["l2a"]: [B,C,H,W]
        """
        x1 = batch["l1c"]
        x2 = batch["l2a"]

        with torch.no_grad():
            e1 = self.image_encoder.get_patch_embeddings(x1)  # [B,P(+1),D]
            e2 = self.image_encoder.get_patch_embeddings(x2)

        # Drop CLS token if present
        if e1.shape[1] == e2.shape[1] and (e1.shape[1] > 1):
            e1 = e1[:, 1:, :]
            e2 = e2[:, 1:, :]

        e1c = self.adapter(e1)  # [B,P,D]

        return e1c, e2

    # -----------------------
    # Training step API
    # -----------------------
    def compute_loss(self, e1c, e2):
        """MSE loss only (cosine optional)."""
        return F.mse_loss(e1c, e2)
