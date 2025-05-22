import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Sequence, Optional, Callable

class CausalConv1d(nn.Module):
    __constants__ = ['_pad']

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding
        )
        self._pad = padding

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        if self._pad > 0:
            return out[..., :-self._pad]
        return out

class TemporalBlock(nn.Module):
    __constants__ = ['downsample']

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        activation: Callable[[], nn.Module]
    ) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=out_ch)
        self.act1 = activation()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=out_ch)
        self.act2 = activation()
        self.drop2 = nn.Dropout(dropout)

        if in_ch != out_ch:
            self.downsample = nn.Conv1d(in_ch, out_ch, 1)
        else:
            self.downsample: Optional[nn.Conv1d] = None
        self.final_act = nn.Mish()

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.act1(out)
        out = self.drop1(out)

        out = self.conv2(x if False else out)
        out = self.gn2(out)
        out = self.act2(out)
        out = self.drop2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.final_act(out + res)

class TemporalConvNet(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_channels: Sequence[int],
        kernel_size: int = 3,
        dropout: float = 0.2,
        activation: Callable[[], nn.Module] = nn.Mish
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for i, out_ch in enumerate(num_channels):
            in_ch = num_inputs if i == 0 else num_channels[i-1]
            dilation = 2 ** i
            layers.append(
                TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout, activation)
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

class TCNGaussian(nn.Module):
    def __init__(
        self,
        input_size: int = 4,
        output_size: int = 3,
        num_channels: Sequence[int] = (32, 32, 64, 64, 128, 128, 256, 256),
        kernel_size: int = 3,
        dropout: float = 0.2,
        activation: Callable[[], nn.Module] = nn.Mish,
        eps: float = 1e-4
    ) -> None:
        super().__init__()
        self.eps = eps
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout, activation)
        self.head = nn.Linear(num_channels[-1], output_size * 2)
        self.softplus = nn.Softplus()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        features = self.tcn(x)
        last = features[..., -1]
        out = self.head(last)
        mu, raw_var = out.chunk(2, dim=1)
        var = self.softplus(raw_var)
        return mu, var

    def loss_function(
        self,
        mu: Tensor,
        var: Tensor,
        target: Tensor
    ) -> Tensor:
        return F.gaussian_nll_loss(mu, target, var, eps=self.eps, full=True)
