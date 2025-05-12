import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

def _init_weights(module: nn.Module) -> None:
    """He‐normal initialization for Conv1d and Linear layers, zero bias."""
    if isinstance(module, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)

class CausalConv1d(nn.Conv1d):
    """
    1D causal convolution with automatic left‐padding and optional weight norm.
    Inherits from nn.Conv1d for simplicity.
    [B, L, C]
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
        padding = (kernel_size - 1) * dilation
        super().__init__(
            in_channels, out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            bias=bias
        )
        if use_weight_norm:
            weight_norm(self)
        self.register_buffer('_pad', torch.tensor(padding))
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv1d already applies symmetric padding; slice off the right side.
        x = super().forward(x)
        if self._pad.item() > 0:
            return x[..., :-self._pad.item()]
        return x

class TemporalBlock(nn.Module):
    """Residual block in a TCN with two causal convs + dropout + activation."""
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
        layers = []
        self.original_in_ch = in_ch
        for _ in range(2):
            layers.append(CausalConv1d(in_ch, out_ch, kernel_size, dilation))
            layers.append(activation())
            layers.append(nn.Dropout(dropout))
            in_ch = out_ch  # for second conv
        self.net = nn.Sequential(*layers)
        # 1×1 downsampling if channels differ
        self.downsample = (
            nn.Conv1d(self.original_in_ch, out_ch, 1)
            if self.original_in_ch != out_ch else None
        )
        if self.downsample:
            self.downsample.apply(_init_weights)
        self.final_act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.final_act(out + res)

class TemporalConvNet(nn.Module):
    """Stack of TemporalBlocks with exponentially increasing dilations."""
    def __init__(
        self,
        num_inputs: int,
        num_channels: int,
        num_levels: int,
        kernel_size: int = 2,
        dropout: float = 0.2,
        activation: type[nn.Module] = nn.ReLU
    ):
        super().__init__()
        blocks = []
        for i in range(num_levels):
            in_ch = num_inputs if i == 0 else num_channels
            dilation = 2 ** i
            blocks.append(
                TemporalBlock(
                    in_ch, num_channels,
                    kernel_size, dilation,
                    dropout, activation
                )
            )
        self.network = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class TCNGaussian(nn.Module):
    """
    Temporal Conv Net that outputs mean and (positive) variance for a Gaussian.
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_channels: int,
        num_levels: int,
        kernel_size: int = 2,
        dropout: float = 0.2,
        activation: type[nn.Module] = nn.ReLU,
        eps: float = 1e-3
    ):
        super().__init__()
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
        # apply He init to linears
        self.linear_mean.apply(_init_weights)
        self.linear_logvar.apply(_init_weights)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, input_size, seq_len)
        Returns:
            mean: (batch, output_size)
            var: (batch, output_size), guaranteed > eps
        """
        features = self.tcn(x)              # (batch, num_channels, seq_len)
        last = features[..., -1]            # (batch, num_channels)
        mean = self.linear_mean(last)       # (batch, output_size)
        logvar = self.linear_logvar(last)   # (batch, output_size)
        var = F.softplus(logvar) + self.eps
        return mean, var
