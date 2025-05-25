import torch
import torchdiffeq as ode
from robot_models.single_track_pacejka import observation
from code_my.filters.ukf_origin import UKF
import numpy as np
from typing import Dict


class UKFModelStepperInference(torch.nn.Module):
    def __init__(self, model, noise_model, dt, state_dim=5,
                 device=torch.device("cpu")) -> None:
        super().__init__()
        self.noise_model = noise_model
        self.model = model
        self.dt = dt
        self.state_dim = state_dim
        self.atol = 1e-6
        self.rtol = 1e-5
        self.method = "rk4"
        self.step_size = dt
        self.backprop_adjoint = False
        
        self.device = device
        self.ukf = UKF(state_dim, meas_dim=4, device=device)
        pass

    def _state_transition_func(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, sigma, state_dim]
        u: [batch, control_dim]
        """
        ur = u.unsqueeze(1).repeat(1, x.shape[1], 1)
        xu = torch.cat((x, ur), dim=-1)
        x_next = ode.odeint(
            self.model,
            xu,
            self.dt,
            rtol=self.rtol,
            atol=self.atol,
            method=self.method,
            options={"step_size": self.step_size},
            backprop_adjoint=self.backprop_adjoint
        )[-1]
        x_next = x_next[..., : self.state_dim]
        return x_next

    def _observation_func(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, sigma, state_dim]
        u: [batch, control_dim]
        """
        ur = u.unsqueeze(1).repeat(1, x.shape[1], 1)
        xu = torch.cat((x, ur), dim=-1)
        return observation(self.model, xu)

    def forward(self, X_hat, P, u, y, X_GT=None):
        """
        X_hat: [batch_size, state_dim]
        P: [batch_size, state_dim, state_dim]
        u: [batch_size, control_dim]
        y: [batch_size, meas_dim]
        X_gt [batch_size, state_dim]
        """

        if X_GT is not None:
            Q, R, _ = self.noise_model(X_GT)
        else:
            Q, R, _ = self.noise_model(X_hat)

        Q = self.project_entropy(Q, lower_bound=self.q_entr_lb)

        X_hat, P = self.ukf.predict(X_hat, P, Q,
                                    self._state_transition_func, u)

        X_hat, P = self.ukf.update(
            X_hat,
            P,
            y,
            R,
            self._observation_func,
            u,
        )

        return X_hat, P, self.calc_normal_entropy(Q), self.calc_normal_entropy(R)

    def P0(self, X):
        _, _, P = self.noise_model(X)
        return P
