import torch
import torchdiffeq as ode
from robot_models.single_track_pacejka import observation
from ukf import UKF
import numpy as np
from typing import Dict

class UKFModelStepperInference(torch.nn.Module):
    def __init__(self, vehicle_model, dt, state_dim=5,
                 device=torch.device("cpu")) -> None:
        super().__init__()
        self.vehicle_model = vehicle_model
        self.dt = dt
        self.state_dim = state_dim
        self.atol = 1e-6
        self.rtol = 1e-5
        self.method = "rk4"
        self.step_size = dt
        self.backprop_adjoint = False
        self.diag_Q = torch.diag(torch.tensor(
            [1e-3, 1e-3, 1e-2, 1e-2, 1e-5], device=device).sqrt())
        self.diag_R = torch.diag(torch.tensor(
            [5e1, 5e1, 1e-1, 1e-1], device=device).sqrt())
        self.diag_P0 = torch.diag(torch.tensor(
            [1e-3, 1e-3, 1e-3, 1e-3, 1e-5], device=device).sqrt())
        
        self.device = device
        self.ukf = UKF(state_dim, meas_dim=4, device=device)
        pass

    def _state_transition_func(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        ur = u.unsqueeze(1).repeat(1, x.shape[1], 1)
        xu = torch.cat((x, ur), dim=-1)
        x_next = ode.odeint(
            self.vehicle_model,
            xu,
            torch.tensor([0, 0.01], device=x.device),
            method="rk4",
            rtol=1e-5,
            atol=1e-6
        )[-1]
        x_next = x_next[..., : self.state_dim]
        return x_next

    def _observation_func(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        ur = u.unsqueeze(1).repeat(1, x.shape[1], 1)
        xu = torch.cat((x, ur), dim=-1)
        return observation(self.vehicle_model, xu)

    def forward(self, X_hat, P, u, y):
        X_hat, P = self.ukf.predict(X_hat, P, self.diag_Q,
                                    self._state_transition_func, u)
        X_hat, P = self.ukf.update(
            X_hat,
            P,
            y,
            self.diag_R,
            self._observation_func,
            u,
        )
        return X_hat, P

