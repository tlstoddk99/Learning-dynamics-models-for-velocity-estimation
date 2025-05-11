import torch
from tire_models.base import BaseTireModel
from utils.state_wrapper import STATE_DEF_LIST, StateWrapper

# import namedtupled
import collections


class NeuralPacejkaTireModelSRSA(BaseTireModel):
    def __init__(self, vehicle_parameters):
        super(NeuralPacejkaTireModelSRSA, self).__init__(vehicle_parameters)

        self.state_to_forces = torch.nn.Sequential(
            torch.nn.Linear(4, 64),
            torch.nn.Sigmoid(),
            torch.nn.Linear(64, 64),
            torch.nn.Sigmoid(),
            torch.nn.Linear(64, 4)
        )

    def forward(self, x):
        wx = StateWrapper(x)
        sa_f = self.slip_angle_front_func(x)
        sa_r = self.slip_angle_rear_func(x)
        sr = self.slip_ratio_func(x)
        sr_front = self.slip_ratio_front_func(x)
        nn_input = torch.stack([sa_f, sa_r, sr, sr_front], dim=-1)
        F = self.state_to_forces(nn_input) * wx.friction.unsqueeze(-1)
        Fy_f_, Fy_r_, Fx_f_, Fx_r_ = F.unbind(dim=-1)

        Fy_f_ = -1.0 * \
            torch.nn.functional.softplus(
                Fy_f_) * torch.nn.functional.tanh(100*sa_f)

        Fy_r_ = -1.0 * \
            torch.nn.functional.softplus(
                Fy_r_) * torch.nn.functional.tanh(100*sa_r)

        Fx_r_ = torch.nn.functional.softplus(
            Fx_r_) * torch.nn.functional.tanh(100*sr)

        return torch.stack([Fy_f_, Fy_r_, Fx_f_, Fx_r_], dim=-1)
