"""ViT Models Implementation with or without factorization for SITS and only factorization for MIX and ENV"""

import copy
import math
from typing import List, Optional, Tuple

import timm
import torch
import torch.nn.functional as F
from einops import repeat
from peft import LoraConfig, TaskType, get_peft_model
from timm.layers import Mlp, PatchEmbed
from timm.models import VisionTransformer
from torch import nn

from src.constants import ORDER_INDEX
from src.models.convlstm import ConvLSTM, TabLSTM
from src.models.fpn_decoder import FPNDecoder
from src.utils.torch_utils import initialize_weights_block, interpolate_pos_embed_mod


def vit_backbone_factory(vit_type: str, pretrained: bool = True, **kwargs) -> VisionTransformer:
    """Factory function for ViT backbone"""

    if vit_type == "vit_small_patch16_224":
        return timm.create_model(
            "vit_small_patch16_224.augreg2_in21k_ft_in1k", pretrained=pretrained, num_classes=0, global_pool=""
        )
    if vit_type == "vit_base_patch16_224":
        return timm.create_model(
            "vit_base_patch16_224.augreg2_in21k_ft_in1k", pretrained=pretrained, num_classes=0, global_pool=""
        )
    if vit_type == "vit_large_patch16_224":
        return timm.create_model(
            "vit_large_patch16_224.augreg2_in21k_ft_in1k", pretrained=pretrained, num_classes=0, global_pool=""
        )
    if vit_type == "vit_huge_patch16_224":
        return timm.create_model(
            "vit_huge_patch16_224.augreg2_in21k_ft_in1k", pretrained=pretrained, num_classes=0, global_pool=""
        )
    if vit_type == "dino_small":
        return timm.create_model(
            "vit_small_patch8_224.dino", pretrained=pretrained, num_classes=0, global_pool="", **kwargs
        )
    if vit_type == "dinov2_small":
        return timm.create_model(
            "vit_small_patch14_dinov2.lvd142m", pretrained=pretrained, num_classes=0, global_pool="", **kwargs
        )
    if vit_type == "dinov2_base":
        return timm.create_model(
            "vit_base_patch14_dinov2.lvd142m", pretrained=pretrained, num_classes=0, global_pool="", **kwargs
        )
    if vit_type == "dinov2_large":
        return timm.create_model(
            "vit_large_patch14_dinov2.lvd142m", pretrained=pretrained, num_classes=0, global_pool="", **kwargs
        )
    if vit_type == "dinov2_giant":
        return timm.create_model(
            "vit_giant_patch14_dinov2.lvd142m", pretrained=pretrained, num_classes=0, global_pool="", **kwargs
        )

    raise ValueError(f"Unknown vit type: {vit_type}")


class CustomPatchEmbed(nn.Module):
    """Custom Patch Embedding with Temporal and Spatial Embedding"""

    def __init__(self, patch_embed: PatchEmbed, proj: Mlp, flag_facto: bool = False):
        super().__init__()
        self.patch_embed = patch_embed
        self.proj = proj
        self.flag_facto = flag_facto

    # Reminder in the augmentation the DOY is normalized by 366
    def _temporal_embedding(self, x: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:

        B, T = xt.shape
        _, num_patches, _ = x.shape

        xt_sin = torch.sin(2 * torch.tensor(math.pi) * xt)
        xt_cos = torch.cos(2 * torch.tensor(math.pi) * xt)
        xt_sin = repeat(xt_sin, "b t -> b t p", p=num_patches).reshape(B * T, num_patches, 1)
        xt_cos = repeat(xt_cos, "b t -> b t p", p=num_patches).reshape(B * T, num_patches, 1)

        xt = torch.cat([xt_sin, xt_cos], dim=-1)

        return xt

    def _spatial_embedding(self, x: torch.Tensor, xpos: torch.Tensor) -> torch.Tensor:
        B, T, _, H, W = xpos.shape
        _, num_patches, _ = x.shape
        xpos = nn.AvgPool2d(kernel_size=self.patch_embed.patch_size)(xpos.reshape(B * T, 2, H, W))
        xpos = xpos.reshape(-1, 2, num_patches)
        xpos = torch.stack(
            [torch.sin(xpos[:, 0, :]), torch.cos(xpos[:, 0, :]), torch.sin(xpos[:, 1, :]), torch.cos(xpos[:, 1, :])],
            dim=1,
        )
        xpos = xpos.permute(0, 2, 1)

        return xpos

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward call of the layer"""

        B, T, C, H, W = x.shape

        # Extract Temporal and Positional Information
        xt = x[:, :, -1, 0, 0]
        xpos = x[:, :, -3:-1, :, :]
        x = x[:, :, :-3]

        # Compute Patch Embedding
        x = x.reshape(B * T, C - 3, H, W)
        x = self.patch_embed(x)
        _, num_patches, embed_dim = x.shape

        # Compute Temporal Embedding and Spatial Embedding -> Output is B * T, N, 6
        xt = self._temporal_embedding(x, xt)
        xpos = self._spatial_embedding(x, xpos)
        xemb = torch.cat([xt, xpos], dim=-1)

        # Project to Embedding Dimension
        xemb = self.proj(xemb)
        x = x + xemb

        # Rest of forward_features from ViT
        if self.flag_facto:
            x = x.reshape(B * T, num_patches, embed_dim)
        else:
            x = x.reshape(B, num_patches * T, embed_dim)

        return x


class ViTModel(nn.Module):
    """Non Factorize SITS Only ViT Model."""

    def __init__(
        self,
        vit_type: str,
        pretrained: bool = True,
        img_res: int = 240,
        patch_size: int = 10,
        input_dim: int = 14,
        num_classes: int = 2,
        out_H: Optional[int] = None,
        out_W: Optional[int] = None,
        flag_lora: bool = False,
        flag_decoder: bool = False,
        flag_enc: str = "loc_doy",
        **kwargs,
    ):
        super().__init__()
        org_backbone = vit_backbone_factory(vit_type, pretrained)
        bias = True if org_backbone.patch_embed.proj.bias is not None else False
        new_patch_embed = PatchEmbed(
            img_size=img_res,
            patch_size=patch_size,
            in_chans=input_dim,
            embed_dim=org_backbone.embed_dim,
            norm_layer=None,
            flatten=org_backbone.patch_embed.flatten,
            bias=bias,
        )
        new_patch_embed.norm = org_backbone.patch_embed.norm
        if (
            org_backbone.patch_embed.patch_size[0] == patch_size
            and org_backbone.patch_embed.patch_size[1] == patch_size
        ):
            new_patch_embed.proj.weight.data[:, :3, :, :] = (
                org_backbone.patch_embed.proj.weight.data
            )  # Copy RGB weights

        if flag_enc == "loc_doy":
            org_backbone.patch_embed = CustomPatchEmbed(
                patch_embed=new_patch_embed,
                proj=Mlp(in_features=6, out_features=org_backbone.embed_dim, act_layer=nn.GELU),
            )
            org_backbone.pos_embed = None

        elif flag_enc == "loc":
            raise NotImplementedError("Need to implement location encoder only")

        else:
            org_backbone.patch_embed = new_patch_embed
            org_backbone = interpolate_pos_embed_mod(org_backbone)

        # Update reduction
        depth = len(org_backbone.blocks)
        reduction = (
            org_backbone.patch_embed.feat_ratio() if hasattr(org_backbone.patch_embed, "feat_ratio") else patch_size
        )
        org_backbone.feature_info = [
            dict(module=f"blocks.{i}", num_chs=org_backbone.embed_dim, reduction=reduction) for i in range(depth)
        ]

        self.features = org_backbone

        # Lora Adaptation
        if flag_lora:
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=kwargs["rank_lora"],
                lora_alpha=kwargs["alpha_lora"],
                lora_dropout=kwargs["dropout_lora"],
                bias="none",
                modules_to_save=["patch_embed.proj"],
                target_modules=["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"],
            )
            self.features = get_peft_model(self.features, lora_config)

        # Decoder
        if flag_decoder:
            self.decoder = FPNDecoder(
                in_channels=org_backbone.embed_dim,
                out_channels=256,
            )
            linear_dim = 256 // 8
            self.indexes = [2, 5, 8, 11]
        else:
            self.decoder = None
            linear_dim = org_backbone.embed_dim

        # Classification Head
        self.head = nn.Linear(linear_dim, num_classes)
        self.out_H = out_H
        self.out_W = out_W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward call of the model"""

        x = x.permute(0, 1, 4, 2, 3)
        B, T, C, H, W = x.shape

        # Forward Features
        if self.features.pos_embed is not None:
            x = x.reshape(B * T, C, H, W)
            x = self.features.patch_embed(x)
            x = self.features._pos_embed(x)
            _, num_patches, embed_dim = x.shape
            x = x.reshape(B, num_patches * T, embed_dim)
        else:
            x = self.features.patch_embed(x)

        x = self.features.patch_drop(x)
        x = self.features.norm_pre(x)

        if self.decoder is not None:
            features = []
            for k, block in enumerate(self.features.blocks):
                x = block(x)
                if k in self.indexes:

                    _, temp_num_patches, embed_dim = x.shape
                    num_patches = temp_num_patches // T

                    # Temporal Average Pooling
                    feat = x.clone()
                    feat = feat.reshape(B, T, num_patches, embed_dim)
                    feat = feat.mean(dim=1)
                    feat = feat.reshape(
                        B,
                        self.features.patch_embed.patch_embed.grid_size[0],
                        self.features.patch_embed.patch_embed.grid_size[1],
                        embed_dim,
                    )
                    feat = feat.permute(0, 3, 1, 2)
                    features.append(feat)

            x = self.decoder(features)
            x = x.permute(0, 2, 3, 1)

        else:
            x = self.features.blocks(x)
            x = self.features.norm(x)
            _, temp_num_patches, embed_dim = x.shape
            num_patches = temp_num_patches // T

            # Temporal Average Pooling
            x = x.reshape(B, T, num_patches, embed_dim)
            if self.features.pos_embed is not None:  # Take out class token
                x = x[:, :, 1:, :]
                x = x.mean(dim=1)
                x = x.reshape(
                    B, self.features.patch_embed.grid_size[0], self.features.patch_embed.grid_size[1], embed_dim
                )
            else:
                x = x.mean(dim=1)
                x = x.reshape(
                    B,
                    self.features.patch_embed.patch_embed.grid_size[0],
                    self.features.patch_embed.patch_embed.grid_size[1],
                    embed_dim,
                )

        # Need B, H, W, C -> B, C, H, W
        x = x.permute(0, 3, 1, 2)
        if self.out_H is not None and self.out_W is not None:
            x = nn.functional.interpolate(x, size=(self.out_H, self.out_W), mode="bilinear")

        # Classification Head Needs B, C, H, W -> B, H, W, C
        x = x.permute(0, 2, 3, 1)
        x = self.head(x)

        # Output is B, num_classes, H, W
        x = x.permute(0, 3, 1, 2)

        return x


class ViTFactorizeModel(nn.Module):
    """Factorized Temporally version of the ViT model for SITS."""

    def __init__(
        self,
        vit_type: str,
        pretrained: bool = True,
        img_res: int = 240,
        patch_size: int = 10,
        input_dim: int = 14,
        num_classes: int = 2,
        out_H: Optional[int] = None,
        out_W: Optional[int] = None,
        flag_lora: bool = False,
        flag_decoder: bool = False,
        doy_int_type: str = "sum",
        temp_enc_type: str = "mean",
        kernel_size: Optional[List[int]] = None,
        n_stack_layers: Optional[int] = None,
        temp_depth: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()

        if isinstance(kernel_size, list):
            kernel_size = tuple(kernel_size)

        self.flag_lora = flag_lora
        self.features, self.spectral_proj, self.flag_order = ViTFactorizeModel._build_encoder(
            vit_type=vit_type,
            pretrained=pretrained,
            img_res=img_res,
            patch_size=patch_size,
            input_dim=input_dim,
            doy_int_type=doy_int_type,
            flag_lora=flag_lora,
            **kwargs,
        )

        # Define DOY integration and Temporal Encoder, TODO: Test without DOY too
        self.doy_int_type = doy_int_type
        if doy_int_type == "sum":
            self.temp_proj = Mlp(
                in_features=2, out_features=self.features.embed_dim, act_layer=nn.GELU
            )  # Maybe too big
        elif doy_int_type == "mean":
            self.temp_proj = Mlp(
                in_features=self.features.embed_dim + 1, out_features=self.features.embed_dim, act_layer=nn.GELU
            )

        self.temp_enc_type = temp_enc_type
        if temp_enc_type == "convlstm":
            self.temp_enc = ConvLSTM(
                self.features.embed_dim,
                self.features.embed_dim,
                kernel_size,
                num_layers=n_stack_layers,
                bias=False,
                batch_first=True,
                return_all_layers=False,
            )
        elif temp_enc_type == "attention":
            self.temp_enc = nn.Sequential(
                *[copy.deepcopy(self.features.blocks[0]).apply(initialize_weights_block) for i in range(temp_depth)]
            )

        # Decoder
        if flag_decoder:
            raise NotImplementedError

        self.decoder = None
        linear_dim = self.features.embed_dim

        # Classification Head
        self.head = nn.Linear(linear_dim, num_classes)
        self.out_H = out_H
        self.out_W = out_W

    @staticmethod
    def _build_encoder(
        vit_type: str,
        pretrained: bool,
        img_res: int,
        patch_size: int,
        input_dim: int,
        doy_int_type: str,
        flag_lora: bool,
        **kwargs,
    ) -> Tuple[nn.Module, Optional[nn.Module], bool]:

        spectral_proj = None
        flag_order = False
        drop_kwargs = {key: item for key, item in kwargs.items() if "drop" in key and key != "dropout_lora"}
        org_backbone = vit_backbone_factory(vit_type, pretrained, **drop_kwargs)
        bias = True if org_backbone.patch_embed.proj.bias is not None else False
        new_patch_embed = PatchEmbed(
            img_size=img_res,
            patch_size=patch_size,
            in_chans=input_dim,
            embed_dim=org_backbone.embed_dim,
            norm_layer=None,
            flatten=org_backbone.patch_embed.flatten,
            bias=bias,
        )
        new_patch_embed.norm = org_backbone.patch_embed.norm

        old_patch_size = org_backbone.patch_embed.patch_size[0]
        new_patch_size = patch_size

        if old_patch_size == new_patch_size:
            new_patch_embed.proj.weight.data[:, :3, :, :] = org_backbone.patch_embed.proj.weight.data
        else:
            old_weights = org_backbone.patch_embed.proj.weight.data
            new_weights = F.interpolate(
                old_weights, size=(new_patch_size, new_patch_size), mode="bilinear", align_corners=True
            )
            new_patch_embed.proj.weight.data[:, :3, :, :] = new_weights

        org_backbone.patch_embed = new_patch_embed

        # Update reduction
        depth = len(org_backbone.blocks)
        reduction = (
            org_backbone.patch_embed.feat_ratio() if hasattr(org_backbone.patch_embed, "feat_ratio") else patch_size
        )
        org_backbone.feature_info = [
            dict(module=f"blocks.{i}", num_chs=org_backbone.embed_dim, reduction=reduction) for i in range(depth)
        ]

        if doy_int_type == "enc_both":
            org_backbone.patch_embed = CustomPatchEmbed(
                patch_embed=new_patch_embed,
                proj=Mlp(in_features=6, out_features=org_backbone.embed_dim, act_layer=nn.GELU),
                flag_facto=True,
            )
            org_backbone.pos_embed = None

        elif doy_int_type == "enc_loc":
            raise NotImplementedError("Need to implement location encoder only")

        else:
            org_backbone.patch_embed = new_patch_embed
            org_backbone = interpolate_pos_embed_mod(org_backbone)

        features = org_backbone
        if flag_lora:
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=kwargs["rank_lora"],
                lora_alpha=kwargs["alpha_lora"],
                lora_dropout=kwargs["dropout_lora"],
                bias="none",
                modules_to_save=["patch_embed.proj"],
                target_modules=["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"],
            )
            features = get_peft_model(features, lora_config)

        return features, spectral_proj, flag_order

    # Reminder in the augmentation the DOY is normalized by 366
    def _temporal_embedding(self, x: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:

        B, T = xt.shape
        _, num_patches, _ = x.shape

        xt_sin = torch.sin(2 * torch.tensor(math.pi) * xt)
        xt_cos = torch.cos(2 * torch.tensor(math.pi) * xt)
        xt_sin = repeat(xt_sin, "b t -> b t p", p=num_patches).reshape(B * T, num_patches, 1)
        xt_cos = repeat(xt_cos, "b t -> b t p", p=num_patches).reshape(B * T, num_patches, 1)

        xt = torch.cat([xt_sin, xt_cos], dim=-1)

        return xt

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward call of the model"""
        x = x.permute(0, 1, 4, 2, 3)
        B, T, C, H, W = x.shape

        # Extract Temporal and Positional Information
        if self.doy_int_type not in ["channel", "enc_both"]:
            xt = x[:, :, -1, 0, 0]
            x = x[:, :, :-1]
            C = C - 1

        if self.flag_order:
            x = x[:, :, ORDER_INDEX]
            mask = torch.ones(C, dtype=torch.bool)
            mask[10] = False  # Drop Band 10
            x = x[:, :, mask]
            C = C - 1

        # Forward Features
        if self.doy_int_type != "enc_both":  # Otherwise factorization is done after the
            x = x.reshape(B * T, C, H, W)
        if self.flag_lora:
            x = self.features.base_model(x)
        else:
            x = self.features(x)

        # Post-Processing Feature Map
        if self.spectral_proj is not None:
            x = x.reshape(
                B * T,
                self.features.patch_embed.num_patches // self.features.patch_embed.t_grid_size,
                self.features.patch_embed.t_grid_size,
                self.features.embed_dim,
            ).permute(0, 1, 3, 2)
            x = self.spectral_proj(x).squeeze(-1)
            spatial_num_patches = self.features.patch_embed.input_size[1] * self.features.patch_embed.input_size[2]
        else:
            if isinstance(self.features.patch_embed, CustomPatchEmbed):
                spatial_num_patches = self.features.patch_embed.patch_embed.num_patches
            else:
                x = x[:, 1:, :]  # Taking out class token
                spatial_num_patches = self.features.patch_embed.num_patches

        # DOY Integration
        if self.doy_int_type == "sum":
            xt = self._temporal_embedding(x, xt)
            xt = self.temp_proj(xt)
            x = x + xt
            x = x.reshape(B, T, spatial_num_patches, self.features.embed_dim)
        elif self.doy_int_type == "conc":  # x is B*T, N, C | TODO: Not working
            x = x.reshape(B, T, spatial_num_patches, self.features.embed_dim)
            xt = repeat(xt, "b t -> b t p", p=spatial_num_patches).unsqueeze(-1)
            x = torch.cat([x, xt])
            x = x.reshape(B * T * spatial_num_patches, self.features.embed_dim + 1)
            x = self.temp_proj(x)
            x = x.reshape(B, T, spatial_num_patches, self.features.embed_dim)
        else:
            x = x.reshape(B, T, spatial_num_patches, self.features.embed_dim)

        # Extract Grid Size
        if self.doy_int_type == "enc_both":
            H_patch, W_Patch = (
                self.features.patch_embed.patch_embed.grid_size[0],
                self.features.patch_embed.patch_embed.grid_size[1],
            )
        else:
            H_patch, W_Patch = self.features.patch_embed.grid_size[0], self.features.patch_embed.grid_size[1]

        # Temporal Encoding
        if self.temp_enc_type == "mean":
            x = x.mean(dim=1)
            x = x.reshape(B, H_patch, W_Patch, self.features.embed_dim)
            # Need B, H, W, C -> B, C, H, W
            x = x.permute(0, 3, 1, 2)
        elif self.temp_enc_type == "max":
            x = x.max(dim=1)
            x = x.reshape(B, H_patch, W_Patch, self.features.embed_dim)
            # Need B, H, W, C -> B, C, H, W
            x = x.permute(0, 3, 1, 2)
        elif self.temp_enc_type == "convlstm":
            x = x.reshape(B, T, H_patch, W_Patch, self.features.embed_dim)
            x = x.permute(0, 1, 4, 2, 3)
            _, x = self.temp_enc(x)
            x = x[0][0]
        elif self.temp_enc_type == "attention":
            x = x.permute(0, 2, 1, 3)
            x = x.reshape(B * spatial_num_patches, T, self.features.embed_dim)
            x = self.temp_enc(x)
            x = x.reshape(B, spatial_num_patches, T, self.features.embed_dim)
            x = x.mean(dim=2)
            x = x.reshape(B, H_patch, W_Patch, self.features.embed_dim)
            x = x.permute(0, 3, 1, 2)
        else:
            raise NotImplementedError

        if self.out_H is not None and self.out_W is not None:
            x = nn.functional.interpolate(x, size=(self.out_H, self.out_W), mode="bilinear")

        # Classification Head Needs B, C, H, W -> B, H, W, C
        x = x.permute(0, 2, 3, 1)
        x = self.head(x)

        # Output is B, num_classes, H, W
        x = x.permute(0, 3, 1, 2)

        return x


class TabViTFactorizeModel(nn.Module):
    """Mix Env & SITS extension of the ViT factorized model."""

    def __init__(
        self,
        pretrained: bool = True,
        tab_input_dim: int = 6,
        output_dim: int = 64,
        env_stack_layers: int = 1,
        dropout: float = 0.0,
        mask_env: bool = False,
        fusion: str = "late",
        tab_enc: bool = True,
        **sat_kwargs,
    ):
        super().__init__()

        self.mask_env = mask_env
        self.fusion = fusion
        self.sat_model = ViTFactorizeModel(pretrained=pretrained, **sat_kwargs)
        self.sat_model.head = nn.Identity()

        self.tab_lstm = TabLSTM(tab_input_dim, output_dim, env_stack_layers, dropout)
        self.tab_enc = tab_enc
        if tab_enc:
            self.tab_lstm = TabLSTM(tab_input_dim, output_dim, env_stack_layers, dropout)
        else:
            self.tab_lstm = nn.LSTM(tab_input_dim, output_dim, env_stack_layers, batch_first=True, dropout=dropout)

        if self.fusion == "mid":
            raise NotImplementedError
        if self.fusion == "late":
            self.linear_head = nn.Linear(output_dim + self.sat_model.features.embed_dim, sat_kwargs["num_classes"])
        elif self.fusion == "late_large":
            self.linear_head = Mlp(
                in_features=output_dim + self.sat_model.features.embed_dim,
                out_features=sat_kwargs["num_classes"],
                act_layer=nn.GELU,
            )

    def forward(
        self, x, xtab, masktab
    ):  # x here is B, T, H, W, C_sat | xmid, xlow are B, T_env, C_env | m_mid, m_low are B, T_env, C_env - 1
        """Forward call of the model"""

        # Masking
        if self.mask_env:
            masktab = masktab.bool()
            xtab[:, :, :-1][masktab] = 0  # Due to the last channel being the DOY

        # Tab Environment Modality
        if self.tab_enc:
            hn = self.tab_lstm(xtab)  # hn is (B, D)
        else:
            _, (hn, _) = self.tab_lstm(xtab)  # hn is (1, B, D)
            hn = hn.squeeze(0)  # B, D

        hn = hn.unsqueeze(-1).unsqueeze(-1)

        if self.fusion == "mid":
            raise NotImplementedError
        if self.fusion in ["late", "late_large"]:
            sat_out = self.sat_model(x)
            _, _, H_sat, W_sat = sat_out.shape
            hn = hn.repeat(1, 1, H_sat, W_sat)
            sat_out = torch.cat([sat_out, hn], dim=1)
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion}")

        # Classification
        sat_out = sat_out.permute(0, 2, 3, 1)
        sat_out = self.linear_head(sat_out)
        sat_out = sat_out.permute(0, 3, 1, 2)

        return sat_out


class EnvViTFactorizeModel(nn.Module):
    """Env only adaptation of the ViT factorized model."""

    def __init__(
        self,
        env_vit_type: str,
        pretrained: bool = True,
        low_input_dim: int = 10,
        low_input_res: int = 32,
        env_patch_size: int = 8,
        env_stack_layers: int = 1,
        mask_env: bool = False,
        **sat_kwargs,
    ):
        super().__init__()

        self.mask_env = mask_env
        self.mid_model = ViTFactorizeModel(pretrained=pretrained, **sat_kwargs)
        sat_kwargs.pop("vit_type")
        sat_kwargs.pop("img_res")
        sat_kwargs.pop("patch_size")
        sat_kwargs.pop("input_dim")
        self.mid_model.head = nn.Identity()

        self.low_encoder, _, _ = ViTFactorizeModel._build_encoder(
            vit_type=env_vit_type,
            pretrained=pretrained,
            img_res=low_input_res,
            patch_size=env_patch_size,
            input_dim=low_input_dim,
            **sat_kwargs,
        )

        self.low_lstm = nn.LSTM(
            self.low_encoder.embed_dim, self.low_encoder.embed_dim, env_stack_layers, batch_first=True
        )
        self.linear_head = nn.Linear(
            self.mid_model.features.embed_dim + self.low_encoder.embed_dim, sat_kwargs["num_classes"]
        )

    def forward(
        self, xmid, xlow, m_mid, m_low
    ):  # xmid, xlow are B, T_env, H, W, C_env | m_mid, m_low are B, T_env, H, W, C_env - 1
        """Forward call of the model"""

        # Masking
        if self.mask_env:
            m_mid = m_mid.bool()
            xmid[:, :, :, :, :-1][m_mid] = 0  # Due to the last channel being the DOY
            m_low = m_low.bool()
            xlow[:, :, :, :, :-1][m_low] = 0  # Due to the last channel being the DOY

        # Middle Modality
        mid_out = self.mid_model(xmid)
        _, _, H_mid, W_mid = mid_out.shape

        # Low Environment Modality
        xlow = xlow.permute(0, 1, 4, 2, 3)
        B, T_env, C_low, H_low, W_low = xlow.shape
        xlow = xlow.reshape(B * T_env, C_low, H_low, W_low)
        low_features = self.low_encoder.base_model(xlow) if self.mid_model.flag_lora else self.low_encoder(xlow)
        low_features = low_features.mean(dim=1)
        low_features = low_features.reshape(B, T_env, -1)
        _, (low_hn, _) = self.low_lstm(low_features)
        low_hn = low_hn.squeeze(0)

        # Concatenate
        hn = low_hn
        hn = hn.unsqueeze(-1).unsqueeze(-1)
        hn = hn.repeat(1, 1, H_mid, W_mid)
        mid_out = torch.cat([mid_out, hn], dim=1)

        # Classification
        mid_out = mid_out.permute(0, 2, 3, 1)
        mid_out = self.linear_head(mid_out)
        mid_out = mid_out.permute(0, 3, 1, 2)

        return mid_out
