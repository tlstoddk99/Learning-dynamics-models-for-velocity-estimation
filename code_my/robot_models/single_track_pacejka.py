import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F

class SingleTrackPacejkaModel(Module):
    """
    Single-track vehicle model using a Pacejka tire model.
    """

    def __init__(self, vehicle_parameters: Module, tire_model: Module) -> None:
        super().__init__()
        self.p = vehicle_parameters  # type: ignore[attr-defined]
        self.tire_model = tire_model

    def forward(self, t, x: Tensor) -> Tensor:
        # x: [..., 7] = [v_x, v_y, r, omega_wheels, friction, delta, Iq]
        v_x, v_y, r, omega_wheels, friction, delta, Iq = torch.unbind(x, dim=-1)

        # Tire forces
        tire_forces = self.tire_model(x)
        Fy_f, Fy_r, Fx_f, Fx_r = torch.unbind(tire_forces, dim=-1)

        # Drag force
        F_drag = (
            self.p.Cd0 * torch.sign(v_x)
            + self.p.Cd1 * v_x
            + self.p.Cd2 * v_x * v_x
        )

        # Dynamics
        v_x_dot = (
            1.0 / self.p.m
            * (
                Fx_r
                + Fx_f * torch.cos(delta)
                - Fy_f * torch.sin(delta)
                - F_drag
                + self.p.m * v_y * r
            )
        )

        v_y_dot = (
            1.0 / self.p.m
            * (
                Fx_f * torch.sin(delta)
                + Fy_r
                + Fy_f * torch.cos(delta)
                - self.p.m * v_x * r
            )
        )

        r_dot = (
            1.0 / self.p.I_z
            * (
                (Fx_f * torch.sin(delta) + Fy_f * torch.cos(delta)) * self.p.lf
                - Fy_r * self.p.lr
            )
        )

        omega_wheels_dot = (
            self.p.R / self.p.I_e
            * (
                self.p.K_fi * Iq
                - self.p.R * (Fx_f + Fx_r)
                - omega_wheels * self.p.b1
                - torch.sign(omega_wheels) * self.p.b0
            )
        )

        zeros = torch.zeros_like(friction)

        return torch.stack([
            v_x_dot,
            v_y_dot,
            r_dot,
            omega_wheels_dot,
            zeros,
            zeros,
            zeros,
        ], dim=-1)
        
    def loss_function(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(x, y)


def observation(model: Module, xu: Tensor) -> Tensor:
    """
    Compute observation from full state.
    Returns:
        [a_x, a_y, r, omega_wheels]
    """
    x_dot = model(xu)
    v_x, v_y, r, omega_wheels, _, _, _ = torch.unbind(xu, dim=-1)
    v_x_dot, v_y_dot, _, _, _, _, _ = torch.unbind(x_dot, dim=-1)

    a_x = v_x_dot - r * v_y
    a_y = v_y_dot + r * v_x

    return torch.stack([a_x, a_y, r, omega_wheels], dim=-1)