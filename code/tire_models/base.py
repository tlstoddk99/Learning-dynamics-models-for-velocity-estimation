import torch
from utils.state_wrapper import STATE_DEF_LIST, StateWrapper


class BaseTireModel(torch.nn.Module):
    def __init__(self, vehicle_params):
        super(BaseTireModel, self).__init__()
        self.p = vehicle_params

    def slip_angle_front_func(self, x):
        wx = StateWrapper(x)
        return torch.atan((wx.v_y + self.p.lf * wx.r) / (wx.v_x + self.p.eps)) - wx.delta

    def slip_angle_rear_func(self, x):
        wx = StateWrapper(x)
        return torch.atan((wx.v_y - self.p.lr * wx.r) / (wx.v_x + self.p.eps))

    def slip_ratio_func(self, x):
        wx = StateWrapper(x)
        slip_ratio = (wx.omega_wheels - wx.v_x) / \
            (wx.v_x + self.p.eps)
        return slip_ratio

    def slip_ratio_front_func(self, x):
        wx = StateWrapper(x)
        v_front = wx.v_x * \
            torch.cos(wx.delta) + (wx.v_y + wx.r *
                                   self.p.lr) * torch.sin(wx.delta)
        slip_ratio = (wx.omega_wheels - v_front) / \
            (v_front + self.p.eps)
        return slip_ratio
