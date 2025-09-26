"""Model adaptation of TSViT architecture for Downscaling Head and Tabular Extension"""

from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from einops import repeat
from einops.layers.torch import Rearrange
from torch import nn

from DeepSatModels.models.TSViT.TSViTdense import Transformer


class TabProjection(nn.Module):
    """High-dimension projection of Tabular data"""

    def __init__(self, in_channels: List[int], out_dim: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.projections = nn.ModuleList([nn.Linear(1, out_dim) for _ in range(in_channels)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, C = x.shape
        projected_channels = []
        for i in range(C):
            channel_data = x[:, :, i : i + 1]
            projected_channel = self.projections[i](channel_data)
            projected_channels.append(projected_channel)

        return torch.stack(projected_channels, dim=2)  # B, T, C, D


class TabTSViTDown(nn.Module):
    """Extension of TSViT to process tabular data and downscaling head"""

    def __init__(self, model_config: Dict[str, Any]):
        super().__init__()
        self.image_size = model_config["img_res"]
        self.patch_size = model_config["patch_size"]
        self.num_patches_1d = self.image_size // self.patch_size
        self.num_classes = model_config["num_classes"]
        self.num_frames = model_config["max_seq_len"]
        self.tab_mod = model_config["tab_mod"]
        self.dim = model_config["dim"]
        if "temporal_depth" in model_config:
            self.temporal_depth = model_config["temporal_depth"]
        else:
            self.temporal_depth = model_config["depth"]
        if "spatial_depth" in model_config:
            self.spatial_depth = model_config["spatial_depth"]
        else:
            self.spatial_depth = model_config["depth"]
        if "tab_depth" in model_config:
            self.tab_depth = model_config["tab_depth"]
        else:
            self.tab_depth = model_config["depth"]

        self.heads = model_config["heads"]
        self.dim_head = model_config["dim_head"]
        self.dropout = model_config["dropout"]
        self.emb_dropout = model_config["emb_dropout"]
        self.pool = model_config["pool"]
        self.scale_dim = model_config["scale_dim"]
        assert self.pool in {"cls", "mean"}, "pool type must be either cls (cls token) or mean (mean pooling)"
        num_patches = self.num_patches_1d**2
        self.downsample_factor = model_config["downsample_factor"]
        patch_dim = (model_config["num_channels"] - 1) * self.patch_size**2  # -1 is set to exclude time feature
        self.to_patch_embedding = nn.Sequential(
            Rearrange("b t c (h p1) (w p2) -> (b h w) t (p1 p2 c)", p1=self.patch_size, p2=self.patch_size),
            nn.Linear(patch_dim, self.dim),
        )
        self.to_tab_embedding = TabProjection(self.tab_mod, self.dim)
        self.to_temporal_embedding_input = nn.Linear(366, self.dim)
        self.temporal_token = nn.Parameter(torch.randn(1, self.num_classes, self.dim))
        self.tab_transformer = Transformer(
            self.dim, self.tab_depth, self.heads, self.dim_head, self.dim * self.scale_dim, self.dropout
        )
        self.temporal_transformer = Transformer(
            self.dim, self.temporal_depth, self.heads, self.dim_head, self.dim * self.scale_dim, self.dropout
        )
        self.space_pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dim))
        self.space_transformer = Transformer(
            self.dim, self.spatial_depth, self.heads, self.dim_head, self.dim * self.scale_dim, self.dropout
        )
        self.dropout = nn.Dropout(self.emb_dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim), nn.Linear(self.dim, int(self.downsample_factor * self.patch_size) ** 2)
        )

    def forward(self, x: torch.Tensor, xtab: torch.Tensor, masktab: torch.Tensor):  # mask should be B, T_tab, C_tab
        """Forward call for the model"""
        x = x.permute(0, 1, 4, 2, 3)
        B, T, C, H, W = x.shape
        _, T_tab, C_tab = xtab.shape
        H_out, W_out = int(H * self.downsample_factor), int(W * self.downsample_factor)

        xt = x[:, :, -1, 0, 0]
        x = x[:, :, :-1]
        xt = (xt * 365.0001).to(torch.int64)
        xt = F.one_hot(xt, num_classes=366).to(torch.float32)
        xt = xt.reshape(-1, 366)
        temporal_pos_embedding = self.to_temporal_embedding_input(xt).reshape(B, T, self.dim)

        # Extract Tabular Features and Tabular temporal encoding
        xt_tab = xtab[:, :, -1]
        xtab = xtab[:, :, :-1]
        C_tab -= 1
        xt_tab = (xt_tab * 365.0001).to(torch.int64)
        xt_tab = F.one_hot(xt_tab, num_classes=366).to(torch.float32)
        xt_tab = xt_tab.reshape(-1, 366)
        temporal_pos_embedding_tab = self.to_temporal_embedding_input(xt_tab).reshape(B, T_tab, self.dim)

        x = self.to_patch_embedding(x)
        x = x.reshape(B, -1, T, self.dim)
        x += temporal_pos_embedding.unsqueeze(1)
        x = x.reshape(-1, T, self.dim)

        # Tabular Encoder to extract CLS tokens
        xtab = self.to_tab_embedding(xtab)
        xtab += temporal_pos_embedding_tab.unsqueeze(
            2
        )  # B, T, C, D for xtab and B, T, D for temporal_pos_embedding_tab
        xtab = xtab.reshape(-1, T_tab * C_tab, self.dim)
        cls_temporal_tokens = repeat(self.temporal_token, "() N d -> b N d", b=B)
        xtab = torch.cat((cls_temporal_tokens, xtab), dim=1)  # B, T_tab * C_tab + K, D
        masktab = masktab.reshape(-1, T_tab * C_tab)
        masktab = torch.cat(
            (torch.zeros(B, self.num_classes, device=masktab.device), masktab), dim=1
        )  # B, T_tab * C_tab + K
        xtab = self.tab_transformer(xtab, mask=masktab)

        cls_temporal_tokens = xtab[:, : self.num_classes]  # B, K, D
        cls_temporal_tokens = repeat(
            cls_temporal_tokens, "b n d -> (b p) n d", p=self.num_patches_1d**2
        )  # Had to modify this line
        x = torch.cat((cls_temporal_tokens, x), dim=1)
        x = self.temporal_transformer(x)
        x = x[:, : self.num_classes]
        x = (
            x.reshape(B, self.num_patches_1d**2, self.num_classes, self.dim)
            .permute(0, 2, 1, 3)
            .reshape(B * self.num_classes, self.num_patches_1d**2, self.dim)
        )
        x += self.space_pos_embedding  # [:, :, :(n + 1)]
        x = self.dropout(x)
        x = self.space_transformer(x)
        x = self.mlp_head(x.reshape(-1, self.dim))
        x = x.reshape(
            B, self.num_classes, self.num_patches_1d**2, int(self.downsample_factor * self.patch_size) ** 2
        ).permute(0, 2, 3, 1)
        x = x.reshape(B, H_out, W_out, self.num_classes)
        x = x.permute(0, 3, 1, 2)
        return x
