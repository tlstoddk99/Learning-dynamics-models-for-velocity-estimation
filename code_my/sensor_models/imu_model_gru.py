import torch
from torch import nn
import torch.nn.functional as F

class GaussianGRU(nn.Module):
    def __init__(self,
                 input_size=3,
                 hidden_size=64,
                 output_size=3,
                 gru_layers=1,
                 fc_layers=1,
                 activation=nn.ReLU,
                 dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, gru_layers,
                          batch_first=True,
                          dropout=dropout if gru_layers>1 else 0.0)
        act = activation()
        layers = []
        for _ in range(fc_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(act)
        self.fc = nn.Sequential(*layers) if layers else nn.Identity()
        self.mean_head = nn.Linear(hidden_size, output_size)
        self.zeta_head = nn.Sequential(nn.Linear(hidden_size, output_size), nn.Softplus())

    def forward(self, x, hidden=None):
        out, h = self.gru(x, hidden)
        feat = self.fc(F.relu(out))
        mean = self.mean_head(feat)
        zeta = self.zeta_head(feat)
        var = torch.exp(2 * zeta)
        return mean[:, -1, :], var[:, -1, :]

    def loss_function(self, mean, var, target):
        return F.gaussian_nll_loss(mean, target, var)