import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from typing import Tuple, Sequence

class CausalConv1d(nn.Module):
    """
    Causal 1D convolution: automatic left padding, optional weight normalization.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        use_weight_norm: bool = True
    ) -> None:
        super(CausalConv1d, self).__init__()
        padding = (kernel_size - 1) * dilation
        conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding
        )
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
    """
    Residual block: 2 x (CausalConv1d -> activation -> Dropout) + skip + Mish.
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        activation: nn.Module
    ) -> None:
        super(TemporalBlock, self).__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.act1 = activation
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.act2 = activation
        self.drop2 = nn.Dropout(dropout)

        if in_ch != out_ch:
            self.downsample = weight_norm(nn.Conv1d(in_ch, out_ch, 1))
        else:
            self.downsample = None
        self.final_act = nn.Mish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.act1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.act2(out)
        out = self.drop2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.final_act(out + res)

class TemporalConvNet(nn.Module):
    """
    Stack of TemporalBlocks with exponentially increasing dilations.
    Receptive field with kernel_size=3 and 8 levels covers ~511 timesteps.
    """
    def __init__(
        self,
        num_inputs: int,
        num_channels: Sequence[int], 
        kernel_size: int = 3,
        dropout: float = 0.2,
        activation: nn.Module = nn.Mish()
    ) -> None:
        super(TemporalConvNet, self).__init__()
        self.layers = nn.ModuleList()
        for i, out_ch in enumerate(num_channels):
            in_ch = num_inputs if i == 0 else num_channels[i-1]
            dilation = 2 ** i
            block = TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout, type(activation)())
            self.layers.append(block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

class TCNGaussian(nn.Module):
    """
    TemporalConvNet predicting Gaussian parameters (mean, var) at every timestep.
    For seq_len=500 and 8-layer TCN, outputs per-step distributions.
    """
    def __init__(
        self,
        input_size: int = 3,
        output_size: int = 3,
        num_channels: Sequence[int] = (32, 32, 64, 64, 128, 128, 256, 256),
        kernel_size: int = 3,
        dropout: float = 0.2,
        activation: nn.Module = nn.Mish(),
        eps: float = 1e-4
    ) -> None:
        super(TCNGaussian, self).__init__()
        self.eps = eps
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout, activation)
        self.head = weight_norm(nn.Linear(num_channels[-1], output_size * 2))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.tcn(x)            
        last = features[..., -1]      
        out = self.head(last)         
        mu, raw_var = out.chunk(2, dim=1) 
        var = F.softplus(raw_var)
        return mu, var

    def loss_function(self, mu: torch.Tensor, var: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.gaussian_nll_loss(mu, target, var, eps=self.eps, reduction='mean')
