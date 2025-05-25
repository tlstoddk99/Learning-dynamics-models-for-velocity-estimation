import torch


class BaseTireModel(torch.nn.Module):
    def __init__(self, vehicle_params):
        super(BaseTireModel, self).__init__()
        self.p = vehicle_params

    def slip_angle_front_func(self, x):
        v_x, v_y, r, omega_wheels, friction, delta, Iq = torch.unbind(x, dim=-1)
        return torch.atan((v_y + self.p.lf * r) / (v_x + self.p.eps)) - delta

    def slip_angle_rear_func(self, x):
        v_x, v_y, r, omega_wheels, friction, delta, Iq = torch.unbind(x, dim=-1)
        return torch.atan((v_y - self.p.lr * r) / (v_x + self.p.eps))

    def slip_ratio_func(self, x):
        v_x, v_y, r, omega_wheels, friction, delta, Iq = torch.unbind(x, dim=-1)
        slip_ratio = (omega_wheels - v_x) / \
            (v_x + self.p.eps)
        return slip_ratio

    def slip_ratio_front_func(self, x):
        v_x, v_y, r, omega_wheels, friction, delta, Iq = torch.unbind(x, dim=-1)
        v_front = v_x * \
            torch.cos(delta) + (v_y + r * self.p.lr) * torch.sin(delta)
        slip_ratio = (omega_wheels - v_front) / \
            (v_front + self.p.eps)
        return slip_ratio
