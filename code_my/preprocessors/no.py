import torch


class NoPreprocessing(torch.nn.Module):
    def __init__(self, device=torch.device('cpu'), *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return x
