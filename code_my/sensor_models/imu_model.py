import torch
import torch.nn as nn
import torch.nn.functional as F

class ImuModel(nn.Module):
    def __init__(self, input_channels: int = 3, output_dim: int = 3):
        super(ImuModel, self).__init__()

        channels = 16
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(input_channels, channels, kernel_size=3, padding=1),
            nn.SiLU()
        )
        self.channel_conv = nn.Sequential(
            nn.Conv1d(input_channels, channels, kernel_size=1),
            nn.SiLU()
        )
        self.gate_layer = nn.Sequential(
            nn.Conv1d(channels * 2, channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.mlp = nn.Sequential(
            nn.Linear(channels, 32),
            nn.SiLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t_f = self.temporal_conv(x)  # (B, 16, L)
        c_f = self.channel_conv(x)   # (B, 16, L)

        combined = torch.cat([t_f, c_f], dim=1)  # (B, 32, L)
        gate = self.gate_layer(combined)        # (B, 16, L)
        fused = gate * t_f + (1 - gate) * c_f   # (B, 16, L)

        x = self.pool(fused)                    # (B, 16, 1)
        x = x.view(x.size(0), -1)               # (B, 16)
        x = self.mlp(x)                         # (B, output_dim)
        return x
    
    def loss_function(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(x, y)