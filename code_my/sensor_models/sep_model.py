import torch
import torch.nn as nn
import torch.nn.functional as F


class PINNSeparator(nn.Module):
    """
    1D-CNN backbone with three output branches for dynamics, noise, and bias.
    TorchScript-compatible implementation.
    Physics-inspired loss terms:
      - spectral_loss_high_freq: penalize low-frequency energy in noise branch
      - loss_random_walk: enforce smooth, low-frequency drift in bias branch
    """
    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 32,
        f_cut: float = 20.0,
        fs: float = 100.0,
        lambda_hf: float = 0.1,
        lambda_rw: float = 0.01,
    ):
        super(PINNSeparator, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.f_cut = f_cut
        self.fs = fs
        self.lambda_hf = lambda_hf
        self.lambda_rw = lambda_rw

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(16),
            nn.Conv1d(16, hidden_channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_channels),
        )

        # Three output branches
        self.branch_dyn = nn.Conv1d(hidden_channels, in_channels, kernel_size=1)
        self.branch_noise = nn.Conv1d(hidden_channels, in_channels, kernel_size=1)
        self.branch_bias = nn.Conv1d(hidden_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        x_c = x.transpose(1, 2)  # [B, C, T]
        h = self.encoder(x_c)
        y_d = self.branch_dyn(h).transpose(1, 2)
        y_n = self.branch_noise(h).transpose(1, 2)
        y_b = self.branch_bias(h).transpose(1, 2)
        return y_d, y_n, y_b

    @torch.jit.export
    def loss_function(
        self,
        y_d: torch.Tensor,
        y_n: torch.Tensor,
        y_b: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        loss_rec = F.mse_loss(y_d + y_n + y_b, x)
        loss_hf = self._spectral_loss_high_freq(y_n)
        loss_rw = self._loss_random_walk(y_b)
        return loss_rec + self.lambda_hf * loss_hf + self.lambda_rw * loss_rw

    def _spectral_loss_high_freq(self, y_n: torch.Tensor) -> torch.Tensor:
        # y_n: [B, T, C]
        # Compute real FFT along time dim
        Y = torch.fft.rfft(y_n, dim=1)  # [B, F, C]
        # Compute frequency bins
        _, T, _ = y_n.size()
        freqs = torch.fft.rfftfreq(n=T, d=1.0 / self.fs, device=y_n.device)
        # Build mask: 1 for low frequencies to penalize
        mask = (freqs < self.f_cut).to(Y.dtype).view(1, -1, 1)
        # Penalize low-frequency energy
        return (Y.abs().pow(2) * mask).mean()

    def _loss_random_walk(self, y_b: torch.Tensor) -> torch.Tensor:
        # y_b: [B, T, C]
        diffs = y_b[:, 1:, :] - y_b[:, :-1, :]
        return diffs.pow(2).mean()


