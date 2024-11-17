import torch

from utils.state_wrapper import STATE_DEF_LIST, StateWrapper


class ImuOffestAndRotation(torch.nn.Module):
    def __init__(self, device, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.distance = torch.nn.Parameter(torch.tensor(0.06, device=device),
                                           requires_grad=True)
        self.imu_rotation = torch.nn.Parameter(torch.tensor(-0.04, device=device),
                                               requires_grad=True)

    def forward(self, x):
        wx = StateWrapper(x, STATE_DEF_LIST)

        v_y = wx.v_y - self.distance * wx.r
        v_x = wx.v_x

        ax = torch.cos(self.imu_rotation) * wx.ax_imu - \
            torch.sin(self.imu_rotation) * wx.ay_imu
        ay = torch.sin(self.imu_rotation) * wx.ax_imu + \
            torch.cos(self.imu_rotation) * wx.ay_imu

        return torch.stack([v_x, v_y, wx.r, wx.omega_wheels, wx.delta, wx.Iq,
                            ax, ay, wx.r_imu], dim=-1)
