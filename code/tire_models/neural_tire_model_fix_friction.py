import torch
from tire_models.base import BaseTireModel
from utils.state_wrapper import STATE_DEF_LIST, StateWrapper


class NeuralPacejkaTireModelFixFriction(BaseTireModel):
    def __init__(self, vehicle_parameters):
        super(NeuralPacejkaTireModelFixFriction,
              self).__init__(vehicle_parameters)

        self.state_to_forces = torch.nn.Sequential(
            torch.nn.Linear(10, 64),
            torch.nn.Sigmoid(),
            torch.nn.Linear(64, 64),
            torch.nn.Sigmoid(),
            torch.nn.Linear(64, 4)
        )

    def forward(self, x):
        wx = StateWrapper(x)
        sa_f = self.slip_angle_front_func(wx)
        sa_r = self.slip_angle_rear_func(wx)
        sr = self.slip_ratio_func(wx)
        add_state = torch.stack([sa_f, sa_r, sr], dim=-1)
        x_ext = torch.cat([x, add_state], dim=-1)
        F = self.state_to_forces(x_ext) * 0.4
        Fy_f_, Fy_r_, Fx_f_, Fx_r_ = F.unbind(dim=-1)
        return torch.stack([Fy_f_, Fy_r_, Fx_f_, Fx_r_], dim=-1)
