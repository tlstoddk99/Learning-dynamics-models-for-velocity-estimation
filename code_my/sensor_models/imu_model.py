import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Sequence, Callable

class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise + pointwise 1D convolution with causal padding."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        # depthwise convolution
        self.depthwise = nn.Conv1d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            dilation=dilation, padding=padding,
            groups=in_channels, bias=False
        )
        # pointwise convolution
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True)
        self._padding = padding

    def forward(self, input_seq: Tensor) -> Tensor:
        x = self.depthwise(input_seq)
        # chop off extra padding for causal conv
        if self._padding > 0:
            x = x[..., :-self._padding]
        return self.pointwise(x)


class TemporalBlock(nn.Module):
    """A single temporal block using depthwise-separable convs + residual."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        activation: Callable[[], nn.Module]
    ):
        super().__init__()
        # first conv
        self.conv1 = DepthwiseSeparableConv1d(in_channels, out_channels, kernel_size, dilation)
        self.norm1 = nn.LayerNorm(out_channels)
        self.act1 = activation()
        self.drop1 = nn.Dropout2d(dropout)
        # second conv
        self.conv2 = DepthwiseSeparableConv1d(out_channels, out_channels, kernel_size, dilation)
        self.norm2 = nn.LayerNorm(out_channels)
        self.act2 = activation()
        self.drop2 = nn.Dropout2d(dropout)

        # residual projection if channel dims differ
        self.projection = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else None
        )
        self.final_activation = activation()

    def forward(self, input_seq: Tensor) -> Tensor:
        # input_seq: (batch, channels, time)
        out = self.conv1(input_seq)
        out = out.permute(0, 2, 1)      # → (batch, time, channels)
        out = self.norm1(out)
        out = self.act1(out)
        out = out.permute(0, 2, 1)      # → (batch, channels, time)
        out = self.drop1(out)

        out = self.conv2(out)
        out = out.permute(0, 2, 1)
        out = self.norm2(out)
        out = self.act2(out)
        out = self.drop2(out)
        out = out.permute(0, 2, 1)

        residual = input_seq if self.projection is None else self.projection(input_seq)
        return self.final_activation(out + residual)


class GaussianTCNForecaster(nn.Module):
    """
    Temporal Convolutional Network that outputs a Gaussian forecast (mean & variance)
    for each of `forecast_horizon` future steps.
    """
    def __init__(
        self,
        input_size: int = 3,
        target_size: int = 3,
        channel_layers: Sequence[int] = (8, 8, 16),
        kernel_size: int = 3,
        dropout: float = 0.2,
        activation: Callable[[], nn.Module] = nn.ReLU,
        nll_eps: float = 1e-4,
        forecast_horizon: int = 100,
    ):
        super().__init__()
        self.nll_eps = nll_eps
        self.forecast_horizon = forecast_horizon

        # build TCN layers
        tcn_layers = []
        for i, ch in enumerate(channel_layers):
            in_ch = input_size if i == 0 else channel_layers[i-1]
            dilation = 2 ** i
            tcn_layers.append(
                TemporalBlock(
                    in_ch, ch,
                    kernel_size, dilation,
                    dropout, activation
                )
            )
        self.tcn = nn.Sequential(*tcn_layers)

        # head: project last TCN features to 2*(target_size)*horizon
        self.output_head = nn.Linear(channel_layers[-1], target_size * 2 * forecast_horizon)
        self.softplus = nn.Softplus()

    def forward(self, input_seq: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            input_seq: (batch, input_channels, time_steps)
        Returns:
            mean:     (batch, forecast_horizon, target_size)
            variance: (batch, forecast_horizon, target_size)
        """
        tcn_features = self.tcn(input_seq)              # (batch, channels, time)
        last_feature = tcn_features[:, :, -1]           # (batch, channels)
        head_output = self.output_head(last_feature)    # (batch, 2*target_size*horizon)

        # reshape to (batch, horizon, 2, target_size)
        head_output = head_output.view(
            input_seq.size(0),
            self.forecast_horizon,
            2,
            -1
        )
        mean     = head_output[:, :, 0, :]              # (batch, horizon, target_size)
        raw_var  = head_output[:, :, 1, :]
        variance = self.softplus(raw_var)

        return mean, variance

    def gaussian_nll_loss(self, mean: Tensor, variance: Tensor, target: Tensor) -> Tensor:
        """
        Compute the full Gaussian NLL loss:
        0.5 * [log(variance) + (target - mean)^2 / variance] + constant
        """
        return F.gaussian_nll_loss(mean, target, variance, eps=self.nll_eps, full=True)
