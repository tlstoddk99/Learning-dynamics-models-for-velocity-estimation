import torch
from tire_models.base import BaseTireModel
from utils.state_wrapper import STATE_DEF_LIST, StateWrapper


class PacejkaTireModel(BaseTireModel):
    def __init__(self, vehicle_parameters):
        super(PacejkaTireModel, self).__init__(vehicle_parameters)
        self.vhp = vehicle_parameters

        self.B_f = torch.nn.Parameter(torch.tensor(5.94814182))
        self.C_f = torch.nn.Parameter(torch.tensor(2.27540553))
        self.D_f = torch.nn.Parameter(torch.tensor(0.8095089))

        self.B_r = torch.nn.Parameter(torch.tensor(5.94814182))
        self.C_r = torch.nn.Parameter(torch.tensor(2.27540553))
        self.D_r = torch.nn.Parameter(torch.tensor(0.8095089))

        self.long_B = torch.nn.Parameter(torch.tensor(2.8))
        self.long_C = torch.nn.Parameter(torch.tensor(1.2))
        self.long_mu_tire = torch.nn.Parameter(torch.tensor(0.891585168))

    def Fy_f(self, slip_angle):
        return - self.vhp.Fn_f * self.D_f * torch.sin(self.C_f * torch.atan(self.B_f * slip_angle))

    def Fy_r(self, slip_angle):
        return - self.vhp.Fn_r * self.D_r * torch.sin(self.C_r * torch.atan(self.B_r * slip_angle))

    def Fx(self, slip_ratio):
        return self.vhp.m * self.vhp.g * self.long_mu_tire * torch.sin(self.long_C * torch.atan(self.long_B * slip_ratio))

    def forward(self, x):
        wx = StateWrapper(x)

        Fy_f_ = self.Fy_f(self.slip_angle_front_func(x))
        Fy_r_ = self.Fy_r(self.slip_angle_rear_func(x))
        Fx_ = self.Fx(self.slip_ratio_func(x))

        # Limiting friction circle
        Fx_f_ = Fx_ * self.vhp.lr / (self.vhp.lf + self.vhp.lr)
        Fx_r_ = Fx_ * self.vhp.lf / (self.vhp.lf + self.vhp.lr)

        # Fy_f_2 = Fy_f_ * torch.sqrt(1.0 - (Fx_f_ / (self.vhp.mu_static * self.vhp.Fn_f))**2)
        # Fy_r_2 = Fy_r_ * torch.sqrt(1.0 - (Fx_r_ / (self.vhp.mu_static * self.vhp.Fn_r))**2)

        return torch.stack([Fy_f_, Fy_r_, Fx_f_, Fx_r_], dim=-1)
