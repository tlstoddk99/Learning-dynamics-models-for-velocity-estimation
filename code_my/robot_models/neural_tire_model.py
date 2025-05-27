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

class NeuralPacejkaTireModel(BaseTireModel):
    def __init__(self, vehicle_parameters):
        super(NeuralPacejkaTireModel, self).__init__(vehicle_parameters)

        self.state_to_forces = torch.nn.Sequential(
            torch.nn.Linear(9, 64),
            torch.nn.Sigmoid(),
            torch.nn.Linear(64, 64),
            torch.nn.Sigmoid(),
            torch.nn.Linear(64, 4)
        )

    def forward(self, x):
        v_x, v_y, r, omega_wheels, friction, delta, Iq = torch.unbind(x, dim=-1)
        
        sa_f = self.slip_angle_front_func(x)
        sa_r = self.slip_angle_rear_func(x)
        sr = self.slip_ratio_func(x)
        add_state = torch.stack([sa_f, sa_r, sr], dim=-1)
        x_ext = torch.cat([x[..., :-1], add_state], dim=-1)
        F = self.state_to_forces(x_ext) * friction.unsqueeze(-1)
        Fy_f_, Fy_r_, Fx_f_, Fx_r_ = F.unbind(dim=-1)
        Fy_f_ = -1.0 * torch.nn.functional.softplus(
            Fy_f_) * torch.nn.functional.tanh(100*sa_f)
        Fy_r_ = -1.0 * torch.nn.functional.softplus(
            Fy_r_) * torch.nn.functional.tanh(100*sa_r)
        return torch.stack([Fy_f_, Fy_r_, Fx_f_, Fx_r_], dim=-1)
