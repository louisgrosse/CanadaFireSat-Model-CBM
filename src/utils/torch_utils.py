"""Utility functions for pytorch training."""

import glob
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from pytorch_lightning import LightningModule
from torch import nn


def load_from_checkpoint(model: LightningModule, checkpoint: Optional[str] = None, device=None):

    assert checkpoint is not None, "no path provided for checkpoint, value is None"
    if os.path.isdir(checkpoint):
        checkpoint = max(glob.iglob(checkpoint + "/*.pth"), key=os.path.getctime)
        print("loading model from %s" % checkpoint)
        model.load_from_checkpoint(checkpoint)
    elif os.path.isfile(checkpoint):
        print("loading model from %s" % checkpoint)
        model.load_from_checkpoint(checkpoint, map_location=device)
    else:
        raise FileNotFoundError("provided checkpoint not found, does not mach any directory or file")

    return checkpoint


def get_trainable_params(
    model: LightningModule, model_type: str, lr_ratio: float, lr: float, mode: str
) -> Union[List[Dict[str, Any]], None]:
    """Provide the list of trainable parameters with the assigned learning rate."""

    if mode == "full":
        trainable_params = list(model.parameters())
        return [{"params": trainable_params, "lr": lr}]

    if mode == "adaptive":

        if model_type == "ResNet":
            encoder_params = list(model.encoder.parameters())
            decoder_params = list(model.decoder.parameters())
            temp_encoder_params = list(model.blocks_conv_lstm.parameters())
            head_params = list(model.linear_head.parameters())

            return [
                {"params": encoder_params, "lr": lr_ratio * lr},
                {"params": decoder_params, "lr": lr},
                {"params": temp_encoder_params, "lr": lr},
                {"params": head_params, "lr": lr},
            ]

        if model_type == "ViT":
            encoder_params = list(model.features.parameters())
            emb_proj = list(model.proj.parameters())
            head_params = list(model.head.parameters())

            return [
                {"params": encoder_params, "lr": lr_ratio * lr},
                {"params": emb_proj, "lr": lr},
                {"params": head_params, "lr": lr},
            ]

        raise NotImplementedError(f"Model {type(model)} not implemented for adaptive learning rate")

    raise ValueError("Invalid mode. Choose between 'full' or 'adaptive'")


def interpolate_pos_embed_mod(
    model: Optional[nn.Module] = None,
    pos_embed: Optional[nn.Parameter] = None,
    new_grid_size: Optional[Tuple[int]] = None,
    with_cls: bool = True,
) -> Union[nn.Module, nn.Parameter]:
    """Interpolate the Position Enbedding to a new target size."""

    if model is not None:
        pos_embed = model.pos_embed

    emb_dim = pos_embed.shape[-1]

    cls_token = pos_embed[:, :1, :]
    pos_token = pos_embed[:, 1:, :]

    old_grid_size = (int(pos_token.shape[1] ** 0.5), int(pos_token.shape[1] ** 0.5))

    if model is not None:
        new_grid_size = model.patch_embed.grid_size

    pos_token = pos_token.reshape(1, old_grid_size[0], old_grid_size[1], emb_dim)
    pos_token = pos_token.permute(0, 3, 1, 2)
    pos_token = torch.nn.functional.interpolate(pos_token, size=new_grid_size, mode="bicubic", align_corners=False)
    pos_token = pos_token.permute(0, 2, 3, 1).flatten(1, 2)

    out_pos_emebed = torch.cat([cls_token, pos_token], dim=1) if with_cls else pos_token

    if model is not None:
        model.pos_embed = nn.Parameter(out_pos_emebed)
        return model

    return nn.Parameter(out_pos_emebed)


def get_alpha(current_epoch: int, alpha_max: float = 0.9, factor: float = 0.8) -> float:
    return alpha_max * (factor**current_epoch)


def initialize_weights_block(module: nn.Module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)  # Xavier initialization for Linear layers
        if module.bias is not None:
            nn.init.zeros_(module.bias)  # Initialize biases to zero
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)  # Initialize LayerNorm weights to 1
        nn.init.zeros_(module.bias)  # Initialize LayerNorm biases to 0
