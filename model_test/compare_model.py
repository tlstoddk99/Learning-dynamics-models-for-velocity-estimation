import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

# -------------------------------
# 1. 데이터셋 생성 (Synthetic)
# -------------------------------
class MLPDataset(Dataset):
    """
    MLP 모델용 데이터셋:
    x[t] -> x[t+1] (단일 state 사용)
    """
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, idx):
        x = self.data[idx]         # 현재 state
        y = self.data[idx + 1]     # 다음 state (target)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class SequenceDataset(Dataset):
    """
    GRU, TCN, Informer 모델용 데이터셋:
    이전 seq_length 스텝의 state 버퍼를 입력받아, 그 다음 state를 예측
    """
    def __init__(self, data, seq_length=10):
        super().__init__()
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x_seq = self.data[idx : idx + self.seq_length]  # (seq_length, state_dim)
        y = self.data[idx + self.seq_length]              # next state
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def generate_synthetic_data(num_samples=10000, state_dim=4, noise_std=1, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    data = []
    # 초기 state를 임의로 생성
    state = np.random.randn(state_dim)
    for _ in range(num_samples):
        # 간단한 선형 dynamics: state_{t+1} = A * state_t + noise
        A = np.eye(state_dim) * 0.95  # identity에 가까운 값
        noise = np.random.randn(state_dim) * noise_std
        next_state = A @ state + noise
        data.append(state)
        state = next_state
    return np.array(data)  # (num_samples, state_dim)

# -------------------------------
# 2. 모델 정의
# -------------------------------

# 2.1 MLP 모델 (단일 state 입력)
class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=None):
        super(MLPModel, self).__init__()
        if output_size is None:
            output_size = input_size
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    def forward(self, x):
        # x: (batch_size, input_size)
        return self.net(x)

# 2.2 GRU 모델 (10-step 입력)
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, output_size=None):
        super(GRUModel, self).__init__()
        if output_size is None:
            output_size = input_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.gru(x)  # out: (batch, seq_len, hidden_size)
        out_last = out[:, -1, :]  # 마지막 타임스텝의 출력
        return self.fc(out_last)

# 2.3 TCN 모델 (Causal Conv1d 사용)
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0, dilation=dilation)
    def forward(self, x):
        # x: (batch, channels, seq_length)
        pad = (self.kernel_size - 1) * self.dilation
        x = F.pad(x, (pad, 0))  # 왼쪽 zero pad
        return self.conv(x)

class TCNModel(nn.Module):
    def __init__(self, input_size, num_channels=[64, 64], kernel_size=3, dropout=0.1, output_size=None):
        super(TCNModel, self).__init__()
        if output_size is None:
            output_size = input_size
        layers = []
        in_channels = input_size
        for i, out_channels in enumerate(num_channels):
            dilation = 2 ** i
            layers.append(CausalConv1d(in_channels, out_channels, kernel_size, dilation))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_channels = out_channels
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_size)
    def forward(self, x):
        # x: (batch, seq_len, input_size) -> (batch, channels, seq_len)
        x = x.transpose(1, 2)
        out = self.tcn(x)
        # 마지막 타임스텝의 출력
        out = out[:, :, -1]
        return self.fc(out)

# 2.4 Informer 모델 (TransformerEncoder 기반 단순화 버전)
class InformerModel(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, dropout=0.1, output_size=None):
        super(InformerModel, self).__init__()
        if output_size is None:
            output_size = input_size
        self.linear_in = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_size)
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = self.linear_in(x)  # (batch, seq_len, d_model)
        # Transformer 입력 형식에 맞게 변환 (seq_len, batch, d_model)
        x = x.transpose(0, 1)
        out = self.transformer_encoder(x)
        out_last = out[-1, :, :]  # 마지막 시점의 토큰
        return self.fc(out_last)

# -------------------------------
# 3. 학습 및 평가 도우미 함수
# -------------------------------
def train_model(model, dataloader, optimizer, criterion, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    return avg_loss

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss

# -------------------------------
# 4. 메인 함수: 데이터 생성, 모델 초기화, 학습 및 평가
# -------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 파라미터
    num_samples = 10000
    state_dim = 4
    seq_length = 10      # GRU, TCN, Informer의 입력 버퍼 길이
    batch_size = 64
    epochs = 100
    learning_rate = 1e-4

    # 데이터 생성 및 분할
    data = generate_synthetic_data(num_samples, state_dim)
    train_split = int(0.8 * len(data))
    train_data = data[:train_split]
    test_data = data[train_split:]

    # 데이터셋 준비
    train_dataset_mlp = MLPDataset(train_data)
    test_dataset_mlp  = MLPDataset(test_data)
    train_dataset_seq = SequenceDataset(train_data, seq_length=seq_length)
    test_dataset_seq  = SequenceDataset(test_data, seq_length=seq_length)

    dataloader_train_mlp = DataLoader(train_dataset_mlp, batch_size=batch_size, shuffle=True)
    dataloader_test_mlp  = DataLoader(test_dataset_mlp,  batch_size=batch_size, shuffle=False)
    dataloader_train_seq = DataLoader(train_dataset_seq, batch_size=batch_size, shuffle=True)
    dataloader_test_seq  = DataLoader(test_dataset_seq,  batch_size=batch_size, shuffle=False)

    # 모델 초기화
    models = {
        'MLP': MLPModel(input_size=state_dim).to(device),
        'GRU': GRUModel(input_size=state_dim).to(device),
        'TCN': TCNModel(input_size=state_dim).to(device),
        'Informer': InformerModel(input_size=state_dim).to(device),
    }
    # 각 모델에 맞게 데이터셋 할당
    dataloaders = {
        'MLP': (dataloader_train_mlp, dataloader_test_mlp),
        'GRU': (dataloader_train_seq, dataloader_test_seq),
        'TCN': (dataloader_train_seq, dataloader_test_seq),
        'Informer': (dataloader_train_seq, dataloader_test_seq),
    }

    criterion = nn.MSELoss()
    results = {}
    timings = {}  # 각 모델별 학습 및 평가 시간 저장

    for name, model in models.items():
        print(f"\n=== Training {name} Model ===")
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        train_loader, test_loader = dataloaders[name]

        # 학습 시간 측정
        train_start = time.time()
        train_model(model, train_loader, optimizer, criterion, device, epochs=epochs)
        train_time = time.time() - train_start

        # 평가 (inference) 시간 측정
        eval_start = time.time()
        test_loss = evaluate_model(model, test_loader, criterion, device)
        eval_time = time.time() - eval_start

        results[name] = test_loss
        timings[name] = {'train_time_sec': train_time, 'inference_time_sec': eval_time}
        print(f"{name} Test Loss: {test_loss:.6f}")
        print(f"{name} Training Time: {train_time:.2f} sec, Inference Time: {eval_time:.2f} sec")

    print("\n--- 최종 모델 비교 (Test Loss 및 시간) ---")
    for name in models.keys():
        print(f"{name}: Loss: {results[name]:.6f}, Training Time: {timings[name]['train_time_sec']:.2f} sec, "
              f"Inference Time: {timings[name]['inference_time_sec']:.2f} sec")

if __name__ == '__main__':
    main()
