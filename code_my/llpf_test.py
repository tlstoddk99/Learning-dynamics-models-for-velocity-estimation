import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)


class LearnableGaussianLPF1D(nn.Module):
    """
    Learnable Gaussian Low-Pass Filter for 1D signals.

    - If `learn_kernel` is False, learns a single sigma (std) for the Gaussian.
    - If `learn_kernel` is True, learns the entire kernel weights directly.
    """
    def __init__(self, channels: int, kernel_size: int = 51,
                 init_sigma: float = 1.0, learn_kernel: bool = False):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"

        self.channels = channels
        self.kernel_size = kernel_size
        self.learn_kernel = learn_kernel

        if learn_kernel:
            # Initialize small random weights for direct kernel learning
            self.kernel = nn.Parameter(torch.randn(kernel_size) * 0.01)
        else:
            # Learn log(sigma) for numerical stability
            self.log_sigma = nn.Parameter(torch.log(torch.tensor(init_sigma)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.learn_kernel:
            k = self.kernel
            k = k - k.min()
            k = k / k.sum()
        else:
            sigma = torch.exp(self.log_sigma)
            coords = torch.arange(self.kernel_size,
                                  device=x.device,
                                  dtype=x.dtype) - self.kernel_size // 2
            k = torch.exp(-coords**2 / (2 * sigma**2))
            k = k / k.sum()

        # Prepare for depthwise convolution
        kernel = k.view(1, 1, -1).repeat(self.channels, 1, 1)
        padding = self.kernel_size // 2
        return F.conv1d(x, kernel, padding=padding, groups=self.channels)


def generate_sine_batch(batch_size: int, length: int,
                        freq: float, noise_std: float = 0.5):
    """
    Generate a batch of noisy sine waves and their clean targets.

    Returns:
        noisy (Tensor): shape (batch_size, 1, length)
        clean (Tensor): shape (batch_size, 1, length)
    """
    t = torch.linspace(0, 1, steps=length)
    clean = torch.sin(2 * torch.pi * freq * t).unsqueeze(0)
    clean = clean.expand(batch_size, -1)
    noise = torch.randn_like(clean) * noise_std
    noisy = clean + noise
    return noisy.unsqueeze(1), clean.unsqueeze(1)


if __name__ == '__main__':
    # --- Configuration ---
    set_seed()

    channels = 1
    kernel_size = 31
    init_sigma = 5.0
    learn_kernel = False
    lr = 1e-2
    epochs = 2000
    batch_size = 32
    signal_length = 512
    freq = 5.0

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model, optimizer, loss
    model = LearnableGaussianLPF1D(channels, kernel_size,
                                   init_sigma, learn_kernel).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Training
    loss_history = []
    for epoch in range(1, epochs + 1):
        model.train()
        noisy, clean = generate_sine_batch(batch_size,
                                           signal_length, freq)
        noisy, clean = noisy.to(device), clean.to(device)

        optimizer.zero_grad()
        output = model(noisy)
        loss = criterion(output, clean)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if epoch % 200 == 0:
            info = f'Epoch {epoch}/{epochs} - Loss: {loss:.4f}'
            if not learn_kernel:
                info += f' - Sigma: {torch.exp(model.log_sigma).item():.3f}'
            print(info)

    # Plot loss curve
    plt.figure()
    plt.plot(loss_history)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')

    # Evaluation on a single sample
    model.eval()
    with torch.no_grad():
        noisy_test, clean_test = generate_sine_batch(1,
                                                     signal_length, freq)
        filtered = model(noisy_test.to(device))

    t_axis = np.linspace(0, 1, signal_length)
    noisy_np = noisy_test.squeeze().numpy()
    clean_np = clean_test.squeeze().numpy()
    filtered_np = filtered.cpu().squeeze().numpy()

    plt.figure(figsize=(10, 6))
    plt.plot(t_axis, noisy_np, label='Noisy')
    plt.plot(t_axis, clean_np, label='Clean')
    plt.plot(t_axis, filtered_np, label='Filtered')
    plt.legend()
    plt.title('Signal Denoising')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()

    # Visualize learned kernel or sigma
    if learn_kernel:
        k = model.kernel.detach().cpu().numpy()
        plt.figure()
        plt.plot(k)
        plt.title('Learned Kernel')
        plt.show()
    else:
        print(f'Learned sigma: {torch.exp(model.log_sigma).item():.3f}')
