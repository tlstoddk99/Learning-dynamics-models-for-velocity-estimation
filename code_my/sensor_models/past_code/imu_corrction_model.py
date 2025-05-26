import torch
import torch.nn as nn

class IMUCorrectionNet(torch.nn.Module):
    def __init__(self, input_dim: int = 3, cnn_channels: int = 32, gru_hidden: int = 32):
        super(IMUCorrectionNet, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=cnn_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
       
        self.gru = nn.GRU(input_size=cnn_channels, hidden_size=gru_hidden, num_layers=1, batch_first=True)
        self.fc = nn.Linear(gru_hidden, 3)  # Predict ax_GT, ay_GT, yaw_rate_GT

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [1, seq_len, input_dim]
        x = x.permute(0, 2, 1)  # -> [1, input_dim, seq_len]
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.permute(0, 2, 1)  # -> [1, seq_len, channels]
        
        output, h_n = self.gru(x)  # h_n shape: [1, 1, gru_hidden]
        last_hidden = h_n[0]  # [1, gru_hidden]
        out = self.fc(last_hidden)  # [1, 3]
        return out