"""Script for Multi-Modal ViT Architecture inspired by MultiMAE"""

import copy
from typing import Any, Dict, List, Optional, OrderedDict

import torch
from peft import LoraConfig, TaskType, get_peft_model
from timm.layers import PatchEmbed, trunc_normal_
from torch import nn

from src.constants import TAB_SOURCE_COLS
from src.models.convlstm import ConvLSTM
from src.models.vit import vit_backbone_factory
from src.utils.torch_utils import initialize_weights_block, interpolate_pos_embed_mod


class MultiPatchPatchEmbed(nn.Module):
    """Multi-Modal Patch embedding Layer"""

    def __init__(self, input_adapters: nn.ModuleDict, input_pos: nn.ParameterDict):
        super().__init__()
        self.input_adapters = input_adapters

        for domain in input_adapters:
            if domain not in input_pos:
                raise ValueError(f"Missing Position Embedding for {domain}")

        self.input_pos = input_pos

    def generate_input_info(self):
        """Provide the key information of all modality patch embedding"""
        input_info = OrderedDict()
        i = 0
        for domain, patch_embed in self.input_adapters.items():
            num_patches = patch_embed.num_patches
            d = {
                "num_patches": num_patches,
                "patch_size": patch_embed.patch_size,
                "grid_size": patch_embed.grid_size,
                "start_idx": i,
                "end_idx": i + num_patches,
            }
            i += num_patches
            input_info[domain] = d

        return input_info

    def forward(self, x: Dict[str, torch.Tensor]):
        """Forward call for the layer"""
        input_task_tokens = {
            domain: self.input_adapters[domain](x[domain]) + self.input_pos[domain]
            for domain in self.input_adapters.keys()
        }
        input_tokens = torch.cat([task_tokens for task_tokens in input_task_tokens.values()], dim=1)
        return input_tokens


class MultiViTFactorizeModel(nn.Module):
    """ViT model with multi-modal patch embedding leveraging similar ViT encoder that in MultiMAE"""

    def __init__(
        self,
        vit_type: str,
        pretrained: bool = True,
        num_classes: int = 2,
        out_H: Optional[int] = None,
        out_W: Optional[int] = None,
        flag_lora: bool = False,
        kernel_size: Optional[List[int]] = None,
        n_stack_layers: Optional[int] = None,
        mask_env: bool = False,
        temp_enc_type: str = "convlstm",
        temp_depth: int = 4,
        **kwargs,
    ):
        super().__init__()

        if isinstance(kernel_size, list):
            kernel_size = tuple(kernel_size)

        self.features = MultiViTFactorizeModel._build_encoder(
            vit_type=vit_type, pretrained=pretrained, flag_lora=flag_lora, **kwargs
        )
        self.patch_info = self.features.patch_embed.generate_input_info()
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
        else:
            raise NotImplementedError()
        linear_dim = self.features.embed_dim

        # Classification Head
        self.head = nn.Linear(linear_dim, num_classes)
        self.out_H = out_H
        self.out_W = out_W
        self.mask_env = mask_env

    @staticmethod
    def _domain_adapters(rgb_weights: Optional[torch.Tensor] = None, **kwargs) -> PatchEmbed:
        domain_patch_embed = PatchEmbed(**kwargs)
        if rgb_weights is not None:
            domain_patch_embed.proj.weight.data[:, :3, :, :] = rgb_weights
        return domain_patch_embed

    @staticmethod
    def _build_encoder(
        vit_type: str, pretrained: bool, flag_lora: bool, domain_kwargs: Dict[str, Any], **kwargs
    ) -> nn.Module:
        """Build the ViT encoder with multi-modal patch embedding"""

        drop_kwargs = {key: item for key, item in kwargs.items() if "drop" in key and key != "dropout_lora"}
        org_backbone = vit_backbone_factory(vit_type, pretrained, **drop_kwargs)
        bias = True if org_backbone.patch_embed.proj.bias is not None else False
        norm = org_backbone.patch_embed.norm
        flatten = org_backbone.patch_embed.flatten

        input_adapters = nn.ModuleDict()
        input_pos = nn.ParameterDict()
        for domain, patch_kwargs in domain_kwargs.items():

            if domain == "sat":

                if (
                    org_backbone.patch_embed.patch_size[0] == patch_kwargs["patch_size"]
                    and org_backbone.patch_embed.patch_size[1] == patch_kwargs["patch_size"]
                ):
                    rgb_weights = org_backbone.patch_embed.proj.weight.data
                else:
                    rgb_weights = None

                input_adapters[domain] = MultiViTFactorizeModel._domain_adapters(
                    rgb_weights=rgb_weights,
                    bias=bias,
                    norm_layer=None,
                    flatten=flatten,
                    embed_dim=org_backbone.embed_dim,
                    **patch_kwargs,
                )
                input_adapters[domain].norm = norm
                new_grid_size = input_adapters[domain].grid_size
                pos_embed = org_backbone.pos_embed
                input_pos[domain] = interpolate_pos_embed_mod(
                    pos_embed=pos_embed, new_grid_size=new_grid_size, with_cls=False
                )

            else:
                input_adapters[domain] = MultiViTFactorizeModel._domain_adapters(
                    bias=bias, norm_layer=None, flatten=flatten, embed_dim=org_backbone.embed_dim, **patch_kwargs
                )
                input_adapters[domain].norm = norm
                pos_embed = nn.Parameter(
                    torch.randn(1, input_adapters[domain].num_patches, org_backbone.embed_dim) * 0.02
                )
                pos_embed = trunc_normal_(pos_embed, std=0.02)
                input_pos[domain] = pos_embed

        org_backbone.patch_embed = MultiPatchPatchEmbed(input_adapters=input_adapters, input_pos=input_pos)

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

        return features

    def forward_features(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward call to embedded the features."""
        x = self.features.patch_embed(x)
        x = self.features.patch_drop(x)
        x = self.features.norm_pre(x)
        x = self.features.blocks(x)
        x = self.features.norm(x)
        return x

    def forward(self, x: torch.Tensor, xtab: torch.Tensor, masktab: torch.Tensor) -> torch.Tensor:
        """Forward call of the model"""
        x = x.permute(0, 1, 4, 2, 3)
        B, T, C, H, W = x.shape

        # Masking
        if self.mask_env:
            masktab = masktab.bool()
            xtab[:, :, :-1][masktab] = 0  # Due to the last channel being the DOY

        # Construct domain specific dict based on tab xtab anf sat img: x
        inputs = {"sat": x.reshape(B * T, C, H, W)}
        doy = xtab[:, :, -1]
        k = 0
        for source, cols in TAB_SOURCE_COLS.items():
            inputs[source] = torch.cat(
                [xtab[:, :, k : k + len(cols)].reshape(B * T, -1, 1, 1), doy.reshape(B * T, 1, 1, 1)], dim=1
            )
            k += len(cols)

        # Forward Features
        x = self.forward_features(inputs)

        # Post-Processing Feature Map | TODO: Adapt Extraction
        spatial_num_patches = self.patch_info["sat"]["num_patches"]
        start_idx = self.patch_info["sat"]["start_idx"]
        end_idx = self.patch_info["sat"]["end_idx"]
        x = x[:, start_idx:end_idx, :].reshape(B, T, spatial_num_patches, self.features.embed_dim)

        # Extract Grid Size
        H_patch, W_Patch = self.patch_info["sat"]["grid_size"]

        # Temporal Encoding
        if self.temp_enc_type == "convlstm":
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
            raise NotImplementedError()

        if self.out_H is not None and self.out_W is not None:
            x = nn.functional.interpolate(x, size=(self.out_H, self.out_W), mode="bilinear")

        # Classification Head Needs B, C, H, W -> B, H, W, C
        x = x.permute(0, 2, 3, 1)
        x = self.head(x)

        # Output is B, num_classes, H, W
        x = x.permute(0, 3, 1, 2)

        return x
