import torch

from utils.state_wrapper import STATE_DEF_LIST, StateWrapper


class OrientationPreprocessing(torch.nn.Module):
    def __init__(self, device, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.angle_deviation = torch.nn.Parameter(torch.tensor(-0.05, device=device),
                                                  requires_grad=True)

    def forward(self, x):
        wx = StateWrapper(x, STATE_DEF_LIST)
        v_x = torch.cos(self.angle_deviation) * wx.v_x - \
            torch.sin(self.angle_deviation) * wx.v_y
        v_y = torch.sin(self.angle_deviation) * wx.v_x + \
            torch.cos(self.angle_deviation) * wx.v_y

        return torch.stack([v_x, v_y, wx.r, wx.omega_wheels, wx.delta, wx.Iq,
                            wx.ax_imu, wx.ay_imu, wx.r_imu], dim=-1)
