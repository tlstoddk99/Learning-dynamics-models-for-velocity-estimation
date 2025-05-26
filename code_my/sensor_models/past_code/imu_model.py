import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Callable, Sequence, Tuple, Optional


class DepthwiseSeparableConv1d(nn.Module):
    """
    Depthwise + pointwise 1D causal convolution.
    """
    __constants__ = ['_padding']

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self._padding: int = (kernel_size - 1) * dilation

        # Depthwise convolution (per-channel)
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self._padding,
            groups=in_channels,
            bias=False,
        )
        # Pointwise convolution to mix channels
        self.pointwise = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, channels, time)
        x = self.depthwise(x)
        if self._padding > 0:
            x = x[:, :, :-self._padding]
        return self.pointwise(x)


class TemporalBlock(nn.Module):
    """
    Single TCN block with two depthwise-separable convs, residual, and dropout.
    """
    __constants__ = ['_in_channels', '_out_channels']

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        activation: Callable[[], nn.Module],
    ) -> None:
        super().__init__()
        self._in_channels: int = in_channels
        self._out_channels: int = out_channels

        # First conv + norm + activation + dropout
        self.conv1 = DepthwiseSeparableConv1d(
            in_channels, out_channels, kernel_size, dilation
        )
        self.norm1 = nn.LayerNorm(out_channels)
        self.act1 = activation()
        self.drop1 = nn.Dropout(dropout)

        # Second conv + norm + activation + dropout
        self.conv2 = DepthwiseSeparableConv1d(
            out_channels, out_channels, kernel_size, dilation
        )
        self.norm2 = nn.LayerNorm(out_channels)
        self.act2 = activation()
        self.drop2 = nn.Dropout(dropout)

        # Residual projection if channels differ
        self.projection: Optional[nn.Module]
        if in_channels != out_channels:
            self.projection = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.projection = None

        self.final_activation = activation()

    def forward(self, x: Tensor) -> Tensor:
        # First block
        out = self.conv1(x)
        out = out.permute(0, 2, 1)  # -> (batch, time, channels)
        out = self.norm1(out)
        out = self.act1(out)
        out = out.permute(0, 2, 1)  # -> (batch, channels, time)
        out = self.drop1(out)

        # Second block
        out = self.conv2(out)
        out = out.permute(0, 2, 1)
        out = self.norm2(out)
        out = self.act2(out)
        out = out.permute(0, 2, 1)
        out = self.drop2(out)

        # Residual connection
        res = x if self.projection is None else self.projection(x)
        return self.final_activation(out + res)


class GaussianTCN(nn.Module):
    """
    Temporal Convolutional Network predicting Gaussian parameters.
    Outputs mean and positive variance for each target dimension.
    """

    def __init__(
        self,
        input_size: int = 3,
        target_size: int = 3,
        channel_layers: Sequence[int] = (8, 8, 16),
        kernel_size: int = 3,
        dropout: float = 0.2,
        activation: Callable[[], nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()

        # Build TCN
        layers = []
        for i, ch in enumerate(channel_layers):
            in_ch = input_size if i == 0 else channel_layers[i - 1]
            dilation = 2 ** i
            layers.append(
                TemporalBlock(
                    in_channels=in_ch,
                    out_channels=ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    activation=activation,
                )
            )
        self.tcn = nn.Sequential(*layers)

        # Final head: map last features to 2 * target_size (mean + var)
        self.output_head = nn.Linear(channel_layers[-1], 2 * target_size)
        self.softplus = nn.Softplus()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # x: (batch, input_channels, time_steps)
        features = self.tcn(x)  # -> (batch, channels, time)
        last = features[:, :, -1]  # -> (batch, channels)
        out = self.output_head(last)  # -> (batch, 2 * target_size)

        out = out.view(x.size(0), 2, -1)  # -> (batch, 2, target_size)
        mean = out[:, 0, :]
        var = self.softplus(out[:, 1, :])
        return mean, var

    def loss_function(self, mean: Tensor, variance: Tensor, target: Tensor) -> Tensor:
        return F.gaussian_nll_loss(mean, target, variance)
