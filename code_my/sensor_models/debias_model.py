import torch
import torch.nn as nn
class ResidualBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class IMUDebiasNet(nn.Module):
    def __init__(self, base_channels=64):
        super(IMUDebiasNet, self).__init__()
        # initial convolution
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=base_channels,
                               kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        # one residual block
        self.resblock = ResidualBlock1D(base_channels)
        # global pooling to reduce temporal dimension
        self.pool = nn.AdaptiveAvgPool1d(1)
        # intermediate fully connected
        self.fc = nn.Linear(base_channels, base_channels)
        # output heads
        self.bias_head = nn.Linear(base_channels, 3)
        self.zeta_head = nn.Linear(base_channels, 3)

    def forward(self, x):

        # conv + bn + relu
        out = self.relu(self.bn1(self.conv1(x)))  # [B, base_channels, T]
        # residual block
        out = self.resblock(out)                  # [B, base_channels, T]
        # pool to [B, base_channels, 1] -> [B, base_channels]
        out = self.pool(out).squeeze(-1)
        # fc + relu
        out = self.relu(self.fc(out))             # [B, base_channels]
        # heads
        bias = self.bias_head(out)                # [B, 3]
        zeta = self.zeta_head(out)            # [B, 3]
        # enforce positive covariance diagonal via exp(2*param)
        cov_diag = torch.exp(2 * zeta)      # [B, 3]
        return bias, cov_diag


    def loss_function(self, pred_bias, cov_diag, true_bias):
        term1 = 0.5 * cov_diag.log().sum(dim=1)          # [B]
        term2 = 0.5 * (((pred_bias - true_bias)**2) / cov_diag).sum(dim=1)  # [B]
        return (term1 + term2).mean()
