import torch
import torch.nn as nn

def conv1d_bn_relu(in_channels, out_channels, kernel_size=3, padding=1):
    """Helper: Conv1d + BatchNorm1d + ReLU"""
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True)
    )

class ResidualBlock1D(nn.Module):
    """A single 1D ResNet block: Conv-BN-ReLU-Conv-BN plus skip connection."""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.block(x)
        out = out + x  # residual
        return self.relu(out)

class IMUDebiasNet(nn.Module):
    """
    IMU De-Bias Network (1D ResNet).
    For both accelerometer and gyroscope biases.

    Input: (batch, 3, seq_len)
    Outputs:
      bias    -> (batch, 3)   : estimated bias per axis
      zeta    -> (batch, 3)   : network output ζ, used to compute variance Σ² = exp(2ζ)
    """
    def __init__(self, in_channels=3, hidden_channels=32):
        super().__init__()
        # initial projection
        self.encoder = nn.Sequential(
            conv1d_bn_relu(in_channels, hidden_channels),
            ResidualBlock1D(hidden_channels)
        )
        # global avg pool
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        # bias head
        self.fc_bias = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, in_channels)
        )
        # zeta head
        self.fc_zeta = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, in_channels)
        )

    def forward(self, x):
        # x: [B, 3, T]
        h = self.encoder(x)           # [B, hidden, T]
        h = self.global_pool(h).squeeze(-1)  # [B, hidden]
        bias = self.fc_bias(h)       # [B, 3]
        zeta = self.fc_zeta(h)       # [B, 3]
        return bias, zeta
    
    def loss_function(self, mu, zeta, target):
       # mu, zeta, target: [B, D]
        # variance: Σ² = exp(2ζ)
        sigma2 = torch.exp(2.0 * zeta)             # [B, D]
        # log det Σ² = sum_d log σ²_d = 2 sum_d ζ_d
        logdet = 2.0 * zeta.sum(dim=-1)           # [B]
        # Mahalanobis term: (μ - y)^T Σ^{-1} (μ - y)
        diff = mu - target
        mahal = (diff.pow(2) / sigma2).sum(dim=-1) # [B]
        # per-sample NLL
        nll = 0.5 * (logdet + mahal)
        return nll.mean()
      
      
