"""Script for the ConvLSTM Model and its extension to Tabular data | Code from https://github.com/ndrplz/ConvLSTM_pytorch"""

from typing import List, Optional, Tuple

import torch
from torch import nn

from src.models.tabtsvit import TabProjection


class ConvLSTMCell(nn.Module):
    """Cell for ConvLSTM Model."""

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: Tuple[int, int], bias: bool):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(self, input_tensor: torch.Tensor, cur_state: Tuple[torch.Tensor]) -> torch.Tensor:
        """Forward call of the layer"""
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size: int, image_size: int):
        height, width = image_size
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
        )


class ConvLSTM(nn.Module):
    """ConvLSTM Model"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: Tuple[int, int],
        num_layers: int,
        batch_first: bool = False,
        bias: bool = True,
        return_all_layers: bool = False,
    ):
        """Parameters:
            input_dim: Number of channels in input
            hidden_dim: Number of hidden channels
            kernel_size: Size of kernel in convolutions
            num_layers: Number of LSTM layers stacked on each other
            batch_first: Whether or not dimension 0 is the batch or not
            bias: Bias or no bias in Convolution
            return_all_layers: Return the list of computations for all layers
            Note: Will do same padding.

        Input:
            A tensor of size B, T, C, H, W or T, B, C, H, W
        Output:
            A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
                0 - layer_output_list is the list of lists of length T of each output
                1 - last_state_list is the list of last states
                        each element of the list is a tuple (h, c) for hidden state and memory
        Example:
            >> x = torch.rand((32, 10, 64, 128, 128))
            >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
            >> _, last_states = convlstm(x)
            >> h = last_states[0][0]  # 0 for layer index, 0 for h index
        """

        super().__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError("Inconsistent list length.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(
                ConvLSTMCell(
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=self.kernel_size[i],
                    bias=self.bias,
                )
            )

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor: torch.Tensor, hidden_state=None) -> torch.Tensor:
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()

        # Since the init is done in forward. Can send image size here
        hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size: int, image_size: int):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size: Tuple[int, int]):
        if not (
            isinstance(kernel_size, tuple)
            or (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))
        ):
            raise ValueError("`kernel_size` must be tuple or list of tuples")

    @staticmethod
    def _extend_for_multilayer(param: nn.Parameter, num_layers: int):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class ConvBlock(nn.Module):
    """Convulation Block for CNN architecture"""

    def __init__(
        self,
        in_channels: List[int],
        out_channels: List[int],
        kernel_size: Tuple[int, int],
        stride: int,
        padding: str,
        dropout: float,
    ):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.GroupNorm(1, out_channels)  # Should be equivalent to LayerNorm
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(dropout)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """Forward call of the layer"""
        return self.pool(self.dropout(self.relu(self.norm(self.conv(x)))))


class ConvLSTMNet(nn.Module):
    """SITS only model based on ConvLSTM"""

    def __init__(
        self,
        num_classes: int,
        input_dim: int,
        hidden_dim: int,
        kernel_size: Tuple[int, int],
        n_stack_layers: int,
        n_conv_blocks: int,
        b_kernel_size: Tuple[int, int],
        b_stride: int,
        b_padding: str,
        dropout: float,
        out_H: Optional[int] = None,
        out_W: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()

        if isinstance(kernel_size, list):
            kernel_size = tuple(kernel_size)

        if isinstance(b_kernel_size, list):
            b_kernel_size = tuple(b_kernel_size)

        self.stack_conv_lstm = ConvLSTM(
            hidden_dim,
            hidden_dim,
            kernel_size,
            num_layers=n_stack_layers,
            bias=False,
            batch_first=True,
            return_all_layers=False,
        )

        self.encoder = nn.Sequential(
            *[ConvBlock(input_dim, hidden_dim, b_kernel_size, b_stride, b_padding, dropout)]
            + [ConvBlock(hidden_dim, hidden_dim, b_kernel_size, b_stride, b_padding, dropout)] * (n_conv_blocks - 1)
        )

        self.mlp_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Should be equivalent to LayerNorm
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        self.out_H = out_H
        self.out_W = out_W
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x here is B, T, H, W, C
        """Foward call of the model"""

        # Encoder
        x = x.permute(0, 1, 4, 2, 3)
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        x = self.encoder(x)

        # ConvLSTM
        _, C_enc, H_enc, W_enc = x.shape
        x = x.reshape(B, T, C_enc, H_enc, W_enc)
        _, x = self.stack_conv_lstm(x)
        x = x[0][0]

        # MLP Head
        x = x.permute(0, 2, 3, 1)
        x = self.mlp_head(x)
        x = x.permute(0, 3, 1, 2)
        if self.out_H is not None and self.out_W is not None:
            x = nn.functional.interpolate(x, size=(self.out_H, self.out_W), mode="bilinear")
        return x


class TabLSTM(nn.Module):
    """Tabular Projection and LSTM layer."""

    def __init__(self, input_dim: int, hidden_dim: int, n_stack_layers: int, dropout: float, **kwargs):
        super().__init__()

        self.tab_encoder = TabProjection(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, n_stack_layers, batch_first=True, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x here is B, T, C
        # Tabular Encoder
        x = self.tab_encoder(x)  # B, T, C, D
        x = x.mean(dim=2)  # B, T, D

        # LSTM
        _, (hn, _) = self.lstm(x)  # hn is (1, B, D)

        return hn.squeeze(0)  # B, D


class TabConvLSTMNet(nn.Module):
    """SITS + Tabular Env model leveraging LSTM and ConvLSTM architecture"""

    def __init__(
        self,
        num_classes: int,
        input_dim: int,
        hidden_dim: int,
        kernel_size: Tuple[int, int],
        n_stack_layers: int,
        n_conv_blocks: int,
        b_kernel_size: Tuple[int, int],
        b_stride: int,
        b_padding: str,
        dropout: float,
        tab_mod: int,
        tab_stack_layers: int,
        out_H: Optional[int] = None,
        out_W: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()

        if isinstance(kernel_size, list):
            kernel_size = tuple(kernel_size)

        if isinstance(b_kernel_size, list):
            b_kernel_size = tuple(b_kernel_size)

        self.stack_conv_lstm = ConvLSTM(
            hidden_dim,
            hidden_dim,
            kernel_size,
            num_layers=n_stack_layers,
            bias=False,
            batch_first=True,
            return_all_layers=False,
        )

        self.tab_lstm = TabLSTM(tab_mod + 1, hidden_dim, tab_stack_layers, dropout)

        self.encoder = nn.Sequential(
            *[ConvBlock(input_dim, hidden_dim, b_kernel_size, b_stride, b_padding, dropout)]
            + [ConvBlock(hidden_dim, hidden_dim, b_kernel_size, b_stride, b_padding, dropout)] * (n_conv_blocks - 1)
        )

        self.mlp_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Should be equivalent to LayerNorm
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        self.out_H = out_H
        self.out_W = out_W
        self.num_classes = num_classes

    def forward(self, x, xtab: torch.Tensor, masktab: torch.Tensor):  # x here is B, T, H, W, C
        """Forward call of the model"""

        # Encoder
        x = x.permute(0, 1, 4, 2, 3)
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        x = self.encoder(x)

        # ConvLSTM
        _, C_enc, H_enc, W_enc = x.shape
        x = x.reshape(B, T, C_enc, H_enc, W_enc)
        _, x = self.stack_conv_lstm(x)
        x = x[0][0]

        # Tabular LSTM
        # xtab = torch.cat((xtab, masktab), dim=2) # B, T_tab, C_tab + 1
        xtab = self.tab_lstm(xtab)  # B, D
        xtab = xtab[:, None, None, :].expand(B, H_enc, W_enc, -1)

        # MLP Head
        x = x.permute(0, 2, 3, 1)
        x = torch.cat((x, xtab), dim=3)
        x = self.mlp_head(x)
        x = x.permute(0, 3, 1, 2)
        if self.out_H is not None and self.out_W is not None:
            x = nn.functional.interpolate(x, size=(self.out_H, self.out_W), mode="bilinear")
        return x
