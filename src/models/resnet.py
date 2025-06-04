"""Implementation of ResNet based models for SITS, MIX, and ENV"""

from typing import List, Optional

import timm
import torch
from torch import nn
from torchvision.ops import MLP

from src.models.convlstm import ConvLSTM, TabLSTM
from src.models.unet_decoder import UnetDecoder


def resnet_backbone_factory(resnet_type: str, pretrained: bool = True) -> timm.models.resnet.ResNet:
    """Backbone factory for ResNet"""
    if resnet_type == "resnet18":
        return timm.create_model("resnet18", pretrained=pretrained)
    if resnet_type == "resnet34":
        return timm.create_model("resnet34", pretrained=pretrained)
    if resnet_type == "resnet50":
        return timm.create_model("resnet50", pretrained=pretrained)
    if resnet_type == "resnet101":
        return timm.create_model("resnet101", pretrained=pretrained)
    if resnet_type == "resnet152":
        return timm.create_model("resnet152", pretrained=pretrained)

    raise ValueError(f"Unknown resnet type: {resnet_type}")


class ResNetEncoder(nn.Module):
    """ResNet Encoder with multi-features ouput"""

    def __init__(
        self,
        resnet_type: str,
        pretrained: bool = True,
        input_dim: int = 14,
        keep_bn: bool = True,
        depth: int = 5,
        rgb_conv: bool = True,
    ):
        super().__init__()
        org_backbone = resnet_backbone_factory(resnet_type, pretrained)
        self.features = nn.Sequential(
            nn.Conv2d(input_dim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            (
                org_backbone.bn1
                if keep_bn
                else nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ),
            org_backbone.act1,
            org_backbone.maxpool,
            org_backbone.layer1,
            org_backbone.layer2,
            org_backbone.layer3,
            org_backbone.layer4,
        )

        if rgb_conv:
            first_conv = self.features[0]
            first_conv.weight.data[:, :3, :, :] = org_backbone.conv1.weight.data
            self.features[0] = first_conv

        self.depth = depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward call for the block"""
        features = []
        for k, layer in enumerate(self.features):
            x = layer(x)
            if k >= self.depth and k != 3:
                features.append(x)

        return features


class ResNetConvLSTM(nn.Module):
    """ResNet based model with temporal factorization for SITS"""

    def __init__(
        self,
        resnet_type: str,
        pretrained: bool = True,
        input_dim: int = 14,
        keep_bn: bool = True,
        depth: int = 5,
        encoder_channels: List[int] = [512, 1024, 2048],
        decoder_channels: List[int] = [256, 128, 64],
        kernel_size: List[int] = [3, 3],
        n_stack_layers: int = 1,
        num_classes: int = 2,
        out_H: Optional[int] = None,
        out_W: Optional[int] = None,
        multi_head: bool = False,
        non_spatial: Optional[int] = None,
        rgb_conv: bool = True,
        fusion: Optional[str] = None,
        temp_enc: bool = True,
        **kwargs,
    ):

        super().__init__()

        if isinstance(kernel_size, list):
            kernel_size = tuple(kernel_size)

        self.encoder = ResNetEncoder(resnet_type, pretrained, input_dim, keep_bn, depth, rgb_conv)

        if non_spatial is None and temp_enc:
            self.blocks_conv_lstm = nn.ModuleList(
                [
                    ConvLSTM(
                        in_ch,
                        out_ch,
                        kernel_size,
                        num_layers=n_stack_layers,
                        bias=False,
                        batch_first=True,
                        return_all_layers=False,
                    )
                    for in_ch, out_ch in zip(encoder_channels, decoder_channels[::-1])
                ]
            )
        elif temp_enc:
            self.blocks_conv_lstm = nn.ModuleList(
                [
                    ConvLSTM(
                        in_ch,
                        out_ch,
                        kernel_size,
                        num_layers=n_stack_layers,
                        bias=False,
                        batch_first=True,
                        return_all_layers=False,
                    )
                    for in_ch, out_ch in zip(encoder_channels[:non_spatial], decoder_channels[::-1][:non_spatial])
                ]
                + [
                    nn.LSTM(in_ch, out_ch, n_stack_layers, batch_first=True)
                    for in_ch, out_ch in zip(encoder_channels[non_spatial:], decoder_channels[::-1][non_spatial:])
                ]
            )
        else:
            self.blocks_conv_lstm = None

        unet_encoder_channels = decoder_channels[::-1]
        if fusion == "mid":
            unet_encoder_channels[-1] = unet_encoder_channels[-1] * 2  # Due to the fusion of the environment
        self.decoder = UnetDecoder(
            encoder_depth=depth,
            encoder_channels=unet_encoder_channels,
            decoder_channels=decoder_channels[1:],
            n_blocks=len(decoder_channels) - 1,
            **kwargs,
        )
        self.linear_head = nn.Linear(decoder_channels[-1], num_classes)
        self.out_H = out_H
        self.out_W = out_W
        self.multi_head = multi_head

        if self.multi_head:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear_head_class = nn.Linear(decoder_channels[0], num_classes)

    def forward(self, x: torch.Tensor, xenv: Optional[torch.Tensor] = None) -> torch.Tensor:  # x here is B, T, H, W, C
        """Forward call for the model"""

        # ResNet Encoder
        x = x.permute(0, 1, 4, 2, 3)
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        features = self.encoder(x)

        # ConvLSTM
        if self.blocks_conv_lstm is not None:
            for i, block in enumerate(self.blocks_conv_lstm):
                _, C_enc, H_enc, W_enc = features[i].shape
                features[i] = features[i].reshape(B, T, C_enc, H_enc, W_enc)
                if isinstance(block, nn.LSTM):
                    features[i] = features[i].squeeze(-2).squeeze(-1)
                    _, features[i] = block(features[i])
                    features[i] = features[i][0][0]
                    features[i] = features[i].unsqueeze(-1).unsqueeze(-1)
                else:
                    _, features[i] = block(features[i])
                    features[i] = features[i][0][0]
        else:
            for i in range(len(features)):
                _, C_enc, H_enc, W_enc = features[i].shape
                features[i] = features[i].reshape(B, T, C_enc, H_enc, W_enc)
                features[i] = features[i].mean(axis=1)

        # Halfway fusion
        if xenv is not None:
            _, _, H_enc, W_enc = features[-1].shape
            xenv = xenv.repeat(1, 1, H_enc, W_enc)
            features[-1] = torch.cat([features[-1], xenv], dim=1)

        # Decoder
        x = self.decoder(*features)

        #  Head
        if self.out_H is not None and self.out_W is not None:
            x = nn.functional.interpolate(x, size=(self.out_H, self.out_W), mode="bilinear")
        x = x.permute(0, 2, 3, 1)
        x = self.linear_head(x)
        x = x.permute(0, 3, 1, 2)

        if self.multi_head:
            pool_feat = self.pool(features[-1]).squeeze(-2).squeeze(-1)
            x_class = self.linear_head_class(pool_feat)
            return x, x_class

        return x


class MixResNetConvLSTM(nn.Module):
    """MIX Model with ResNet leveraging spatial environmental predictors"""

    def __init__(
        self,
        env_resnet_type: str,
        pretrained: bool = True,
        keep_bn: bool = True,
        mid_input_dim: int = 6,
        low_input_dim: int = 10,
        output_dim: int = 64,
        env_encoder_channel: int = 512,
        env_stack_layers: int = 1,
        env_depth: int = 7,
        mask_env: bool = False,
        fusion: str = "late",
        **sat_kwargs,
    ):
        super().__init__()

        self.mask_env = mask_env
        self.fusion = fusion
        self.sat_model = ResNetConvLSTM(pretrained=pretrained, keep_bn=keep_bn, fusion=fusion, **sat_kwargs)
        self.sat_model.linear_head = nn.Identity()

        if self.sat_model.multi_head:
            raise NotImplementedError("Multi-head not yet implemented for the Environment Canada model.")

        self.mid_encoder = ResNetEncoder(
            env_resnet_type, pretrained, mid_input_dim, keep_bn, depth=env_depth, rgb_conv=False
        )
        self.low_encoder = ResNetEncoder(
            env_resnet_type, pretrained, low_input_dim, keep_bn, depth=env_depth, rgb_conv=False
        )
        self.mid_lstm = nn.LSTM(env_encoder_channel, output_dim, env_stack_layers, batch_first=True)
        self.low_lstm = nn.LSTM(env_encoder_channel, output_dim, env_stack_layers, batch_first=True)

        if self.fusion == "mid":
            self.linear_head = nn.Linear(int(output_dim / 2), sat_kwargs["num_classes"])
        elif self.fusion == "late":
            self.linear_head = nn.Linear(output_dim * 3, sat_kwargs["num_classes"])
        elif self.fusion == "late_large":
            self.linear_head = MLP(output_dim * 3, hidden_channels=[output_dim, output_dim, sat_kwargs["num_classes"]])

    def forward(
        self, x: torch.Tensor, xmid: torch.Tensor, xlow: torch.Tensor, m_mid: torch.Tensor, m_low: torch.Tensor
    ) -> (
        torch.Tensor
    ):  # x here is B, T, H, W, C_sat | xmid, xlow are B, T_env, H, W, C_env | m_mid, m_low are B, T_env, H, W, C_env - 1
        """Forward call for the model"""

        # Masking
        if self.mask_env:
            m_mid = m_mid.bool()
            xmid[:, :, :, :, :-1][m_mid] = 0  # Due to the last channel being the DOY
            m_low = m_low.bool()
            xlow[:, :, :, :, :-1][m_low] = 0  # Due to the last channel being the DOY

        # Mid Environment Modality
        xmid = xmid.permute(0, 1, 4, 2, 3)
        B, T_env, C_mid, H_mid, W_mid = xmid.shape
        xmid = xmid.reshape(B * T_env, C_mid, H_mid, W_mid)
        mid_features = self.mid_encoder(xmid)[0]
        mid_features = mid_features.reshape(B, T_env, -1)
        _, (mid_hn, _) = self.mid_lstm(mid_features)  # hn is (1, B, D)
        mid_hn = mid_hn.squeeze(0)

        # Low Environment Modality
        xlow = xlow.permute(0, 1, 4, 2, 3)
        B, T_env, C_low, H_low, W_low = xlow.shape
        xlow = xlow.reshape(B * T_env, C_low, H_low, W_low)
        low_features = self.low_encoder(xlow)[0]
        low_features = low_features.reshape(B, T_env, -1)
        _, (low_hn, _) = self.low_lstm(low_features)
        low_hn = low_hn.squeeze(0)

        # Concatenate | Fusion | Satelite Modality
        hn = torch.cat([mid_hn, low_hn], dim=1)
        hn = hn.unsqueeze(-1).unsqueeze(-1)

        if self.fusion == "mid":
            sat_out = self.sat_model(x, xenv=hn)
        elif self.fusion in ["late", "late_large"]:
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


class TabResNetConvLSTM(nn.Module):
    """MIX Model with ResNet leveraging non-spatial environmental predictors"""

    def __init__(
        self,
        pretrained: bool = True,
        keep_bn: bool = True,
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
        self.sat_model = ResNetConvLSTM(pretrained=pretrained, keep_bn=keep_bn, fusion=fusion, **sat_kwargs)
        self.sat_model.linear_head = nn.Identity()

        if self.sat_model.multi_head:
            raise NotImplementedError("Multi-head not yet implemented for the Environment Canada model.")

        self.tab_enc = tab_enc
        if tab_enc:
            self.tab_lstm = TabLSTM(tab_input_dim, output_dim, env_stack_layers, dropout)
        else:
            self.tab_lstm = nn.LSTM(tab_input_dim, output_dim, env_stack_layers, batch_first=True, dropout=dropout)

        if self.fusion == "mid":
            self.linear_head = nn.Linear(sat_kwargs["decoder_channels"][-1], sat_kwargs["num_classes"])
        elif self.fusion == "late":
            self.linear_head = nn.Linear(output_dim + sat_kwargs["decoder_channels"][-1], sat_kwargs["num_classes"])
        elif self.fusion == "late_large":
            self.linear_head = MLP(
                output_dim + sat_kwargs["decoder_channels"][-1],
                hidden_channels=[output_dim, output_dim, sat_kwargs["num_classes"]],
            )

    def forward(
        self, x: torch.Tensor, xtab: torch.Tensor, masktab: torch.Tensor
    ):  # x here is B, T, H, W, C_sat | xmid, xlow are B, T_env, C_env | m_mid, m_low are B, T_env, C_env - 1
        """Forward call for the model"""

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
            sat_out = self.sat_model(x, xenv=hn)
        elif self.fusion in ["late", "late_large"]:
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


class EnvResNetConvLSTM(nn.Module):
    """ENV only Model with ResNet"""

    def __init__(
        self,
        env_resnet_type: str,
        pretrained: bool = True,
        keep_bn: bool = True,
        low_input_dim: int = 10,
        output_dim: int = 64,
        env_encoder_channel: int = 512,
        env_stack_layers: int = 1,
        env_depth: int = 7,
        mask_env: bool = False,
        **sat_kwargs,
    ):
        super().__init__()

        self.mask_env = mask_env
        self.mid_model = ResNetConvLSTM(pretrained=pretrained, keep_bn=keep_bn, rgb_conv=False, **sat_kwargs)
        self.mid_model.linear_head = nn.Identity()

        if self.mid_model.multi_head:
            raise NotImplementedError("Multi-head not yet implemented for the Environment Canada model.")

        self.low_encoder = ResNetEncoder(
            env_resnet_type, pretrained, low_input_dim, keep_bn, depth=env_depth, rgb_conv=False
        )
        self.low_lstm = nn.LSTM(env_encoder_channel, output_dim, env_stack_layers, batch_first=True)
        self.linear_head = nn.Linear(output_dim * 2, sat_kwargs["num_classes"])

    def forward(
        self, xmid: torch.Tensor, xlow: torch.Tensor, m_mid: torch.Tensor, m_low: torch.Tensor
    ) -> torch.Tensor:  # xmid, xlow are B, T_env, H, W, C_env | m_mid, m_low are B, T_env, H, W, C_env - 1
        """Forward call for the model"""

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
        low_features = self.low_encoder(xlow)[0]
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
