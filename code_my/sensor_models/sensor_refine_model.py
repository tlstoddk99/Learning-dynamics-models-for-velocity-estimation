import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

# He-normal initialization for Conv1d and Linear layers
def _init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)

class CausalConv1d(nn.Module):
    """
    Causal 1D convolution: automatic left padding, weight norm after init.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        bias: bool = True,
        use_weight_norm: bool = True
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            bias=bias
        )
        conv.apply(_init_weights)
        if use_weight_norm:
            conv = weight_norm(conv)
        self.conv = conv
        self._pad = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self._pad > 0:
            return out[..., :-self._pad]
        return out

class TemporalBlock(nn.Module):
    """Residual block: 2 x (CausalConv1d -> GN -> ReLU -> Dropout) + skip + ReLU"""
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        activation: type[nn.Module] = nn.ReLU
    ):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(in_ch, out_ch, kernel_size, dilation),
            nn.GroupNorm(min(out_ch, 8), out_ch),
            activation(),
            nn.Dropout(dropout),
            CausalConv1d(out_ch, out_ch, kernel_size, dilation),
            nn.GroupNorm(min(out_ch, 8), out_ch),
            activation(),
            nn.Dropout(dropout),
        )
        self.downsample = (nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None)
        if self.downsample:
            self.downsample.apply(_init_weights)
        self.final_act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.final_act(out + res)

class TemporalConvNet(nn.Module):
    """Stack of TemporalBlocks with exponentially increasing dilations"""
    def __init__(
        self,
        num_inputs: int,
        num_channels: int,
        num_levels: int,
        kernel_size: int = 3,
        dropout: float = 0.2,
        activation: type[nn.Module] = nn.ReLU
    ):
        super().__init__()
        blocks = []
        # Receptive field: R = 1 + (kernel_size - 1)*(2^num_levels - 1)
        for i in range(num_levels):
            in_ch = num_inputs if i == 0 else num_channels
            dilation = 2 ** i
            blocks.append(
                TemporalBlock(in_ch, num_channels, kernel_size, dilation, dropout, activation)
            )
        self.network = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class TCNGaussian(nn.Module):
    """
    TemporalConvNet predicting Gaussian parameters (mean, var) for drift estimation.
    Applies GroupNorm on input channels: [ax, ay] and [r].
    """
    def __init__(
        self,
        input_size: int = 3,
        output_size: int = 3,
        num_channels: int = 64,
        num_levels: int = 8,
        kernel_size: int = 3,
        dropout: float = 0.2,
        activation: type[nn.Module] = nn.ReLU,
        eps: float = 1e-6
    ):
        super().__init__()
        self.eps = eps
        # Input normalization: 2-channel and 1-channel groups
        self.gn_xy = nn.GroupNorm(1, 2)
        self.gn_r  = nn.GroupNorm(1, 1)

        self.tcn = TemporalConvNet(
            num_inputs=input_size,
            num_channels=num_channels,
            num_levels=num_levels,
            kernel_size=kernel_size,
            dropout=dropout,
            activation=activation
        )
        self.linear_mean = nn.Linear(num_channels, output_size)
        self.linear_logvar = nn.Linear(num_channels, output_size)
        self.linear_mean.apply(_init_weights)
        self.linear_logvar.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, 3, seq_len)
        xy = self.gn_xy(x[:, :2, :])
        r  = self.gn_r(x[:, 2:, :])
        x_norm = torch.cat([xy, r], dim=1)

        features = self.tcn(x_norm)         # (B, C, seq)
        last = features[..., -1]             # (B, C)
        mean = self.linear_mean(last)       # (B, out)
        logvar = self.linear_logvar(last)   # (B, out)
        var = torch.exp(logvar).clamp(min=self.eps)
        return mean, var

# Use built-in Gaussian NLL loss
# torch.nn.functional.gaussian_nll_loss(input, target, var, eps, reduction)
def gnll_loss(mean: torch.Tensor, var: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Gaussian Negative Log Likelihood using PyTorch's built-in function.
    """
    return F.gaussian_nll_loss(mean, target, var, eps=eps, reduction='mean')
