import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class MLPGaussian(nn.Module):
    """
    MLP model for Gaussian distribution prediction.
    """
    def __init__(
        self,
        input_size: int = 1500,
        output_size: int = 3,
        num_channels: int = 64,
        num_levels: int = 8,
        activation: type[nn.Module] = nn.ReLU,
        eps: float = 1e-4,
    ):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.num_channels = num_channels
        self.num_levels = num_levels
        self.activation = activation
        self.eps = eps
        modules = nn.ModuleList()
        for i in range(num_levels):
            if i == 0:
                in_ch = input_size
                out_ch = num_channels
                modules.append(nn.Linear(in_ch, out_ch))
                modules.append(activation())
            
            elif i == num_levels - 1:
                in_ch = num_channels
                out_ch = output_size * 2
                modules.append(nn.Linear(in_ch, out_ch))
            else:
                in_ch = num_channels
                out_ch = num_channels
                modules.append(nn.Linear(in_ch, out_ch))
                modules.append(activation())
            
        self.network = nn.Sequential(*modules)
        
        
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: (batch, 3*seq_len)
        Returns:
            mean: (batch, out), var: (batch, out)
        """
        # x: (batch, 3*seq_len)
        # x = x.flatten(start_dim=1)
        
        x = self.network(x)

        mu, raw_var = x.chunk(2, dim=-1)
        
        var = F.softplus(raw_var)+self.eps
        # var = torch.exp(raw_var)
        return mu, var
        
    def loss_function(self, mu: Tensor, var: Tensor, target: Tensor) -> Tensor:
            return F.gaussian_nll_loss(mu, target, var, reduction='mean', full=True)