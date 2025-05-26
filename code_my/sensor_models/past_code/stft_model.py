import torch
import torch.nn as nn

class IMU_BandSplit_STFT_CNNNet(nn.Module):
    """
    STFT 기반 프론트엔드와 2D CNN으로 구성된 경량 IMU 밴드 예측 모델
    (100 Hz, 5분 데이터셋에 최적화)
    """
    def __init__(self, max_freq=50.0, n_fft=128, hop_length=64):
        super(IMU_BandSplit_STFT_CNNNet, self).__init__()
        self.max_freq = max_freq
        self.n_fft = n_fft
        self.hop_length = hop_length

        # 2D CNN 블록 1: 채널 3 → 32
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Dropout(0.2)
        )
        # 2D CNN 블록 2: 채널 32 → 64
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None)),  # freq 축 전역 평균
            nn.Dropout(0.2)
        )
        # Fully-connected 헤드
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 6),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, 3, T)
        B, C, T = x.shape
        # STFT 변환
        specs = []
        for i in range(C):
            stft = torch.stft(x[:, i], n_fft=self.n_fft, hop_length=self.hop_length,
                              return_complex=True)
            specs.append(torch.abs(stft))
        spec = torch.stack(specs, dim=1)  # (B, 3, F, T')

        # CNN 처리
        out = self.conv_block1(spec)  # (B, 32, F/2, T'/2)
        out = self.conv_block2(out)   # (B, 64, 1, T'')

        out = out.mean(dim=-1)        # (B, 64, 1) → (B, 64)
        out = out.squeeze(-1)
        raw = self.fc(out)            # (B, 6)

        # low, ratio → 실제 대역으로 변환
        raw = raw.view(B, 3, 2)
        lows = raw[:, :, 0] * self.max_freq
        ratios = raw[:, :, 1]
        highs = lows + ratios * (self.max_freq - lows)
        bands = torch.stack([lows, highs], dim=2)
        return bands

if __name__ == "__main__":
    dummy = torch.randn(4, 3, 500)
    model = IMU_BandSplit_STFT_CNNNet()
    out = model(dummy)
    print(out.shape)  # (4, 3, 2)
