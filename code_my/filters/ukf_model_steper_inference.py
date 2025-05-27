import torch
import torch.nn as nn
from typing import Tuple
from torch import Tensor
from torch.jit import ignore
import torchdiffeq as ode


class StateTransitionModel(nn.Module):
    def __init__(self, vehicle_model: nn.Module, dt: float, state_dim: int):
        super().__init__()
        self.vehicle_model = vehicle_model
        self.dt = dt
        self.state_dim = state_dim

    @ignore  # TorchScript에서는 odeint를 지원하지 않으므로 무시
    def forward(self, x: Tensor, u: Tensor) -> Tensor:
        ur = u.unsqueeze(1).repeat(1, x.size(1), 1)
        xu = torch.cat((x, ur), dim=-1)
        t = torch.tensor([0.0, self.dt], device=x.device)
        x_next = ode.odeint(self.vehicle_model, xu, t, method="rk4", rtol=1e-5, atol=1e-6)[-1]
        return x_next[..., :self.state_dim]


class MeasurementModel(nn.Module):
    def __init__(self, vehicle_model: nn.Module):
        super().__init__()
        self.vehicle_model = vehicle_model

    @ignore
    def forward(self, x: Tensor, u: Tensor) -> Tensor:
        ur = u.unsqueeze(1).repeat(1, x.size(1), 1)
        xu = torch.cat((x, ur), dim=-1)
        x_dot = self.vehicle_model(xu)

        v_x, v_y, r, omega_wheels, *_ = torch.unbind(xu, dim=-1)
        v_x_dot, v_y_dot, *_ = torch.unbind(x_dot, dim=-1)

        a_x = v_x_dot - r * v_y
        a_y = v_y_dot + r * v_x
        return torch.stack([a_x, a_y, r, omega_wheels], dim=-1)


class UKF(nn.Module):
    def __init__(self, state_dim: int, meas_dim: int, device: torch.device,
                 state_model: nn.Module, meas_model: nn.Module, kappa: float = -2.0):
        super().__init__()
        self.state_dim = state_dim
        self.meas_dim = meas_dim
        self.kappa = kappa
        self.device = device
        self.state_model = state_model
        self.meas_model = meas_model

        n_sigma = 2 * state_dim + 1
        weights = kappa / (state_dim + kappa)
        w_m = torch.full((n_sigma,), 1.0 / (2.0 * (state_dim + kappa)), device=device)
        w_c = w_m.clone()
        w_m[0] = weights
        w_c[0] = weights

        self.register_buffer("weights_m", w_m)
        self.register_buffer("weights_c", w_c)

    def predict(self, X: Tensor, P: Tensor, Q: Tensor, u: Tensor) -> Tuple[Tensor, Tensor]:
        sigma_points = self._generate_sigma_points(X, P)
        propagated = self.state_model(sigma_points, u)
        X_pred, P_pred = self._recover_gaussian(propagated)
        return X_pred, P_pred + Q

    def update(self, X: Tensor, P: Tensor, Z: Tensor, R: Tensor, u: Tensor) -> Tuple[Tensor, Tensor]:
        sigma_points = self._generate_sigma_points(X, P)
        Z_sigma = self.meas_model(sigma_points, u)
        z_pred, Pz = self._recover_gaussian(Z_sigma)
        Pxz = self._cross_covariance(sigma_points, Z_sigma, X, z_pred)

        L = torch.linalg.cholesky(Pz + R)
        I = torch.eye(L.size(-1), device=L.device).expand(L.size(0), -1, -1)
        inv_Pz = torch.cholesky_solve(I, L)

        K = Pxz @ inv_Pz
        innovation = Z - z_pred

        X_upd = X + (K @ innovation.unsqueeze(-1)).squeeze(-1)
        P_upd = P - K @ Pz @ K.transpose(-2, -1)
        return X_upd, P_upd

    def _generate_sigma_points(self, X: Tensor, P: Tensor) -> Tensor:
        sqrt = torch.linalg.cholesky((self.state_dim + self.kappa) * P)
        points = [X]
        for i in range(self.state_dim):
            points.append(X + sqrt[:, :, i])
            points.append(X - sqrt[:, :, i])
        return torch.stack(points, dim=1)

    def _recover_gaussian(self, sigma: Tensor) -> Tuple[Tensor, Tensor]:
        mean = torch.einsum("i,bin->bn", self.weights_m, sigma)
        diff = sigma - mean.unsqueeze(1)
        cov = torch.einsum("i,bin,bis->bns", self.weights_c, diff, diff)
        return mean, cov

    def _cross_covariance(self, X_sigma: Tensor, Z_sigma: Tensor, X_mean: Tensor, Z_mean: Tensor) -> Tensor:
        dx = X_sigma - X_mean.unsqueeze(1)
        dz = Z_sigma - Z_mean.unsqueeze(1)
        return torch.einsum("i,bix,biz->bxz", self.weights_c, dx, dz)


class UKFModelStepperInference(nn.Module):
    def __init__(self, vehicle_model: nn.Module, dt: float, state_dim: int = 5, device=torch.device("cpu")):
        super().__init__()
        self.state_model = StateTransitionModel(vehicle_model, dt, state_dim)
        self.meas_model = MeasurementModel(vehicle_model)
        self.ukf = UKF(state_dim, meas_dim=4, device=device,
                       state_model=self.state_model, meas_model=self.meas_model)

        self.register_buffer("diag_Q", torch.diag(torch.tensor([1e-3, 1e-3, 1e-2, 1e-2, 1e-5], device=device)))
        self.register_buffer("diag_R", torch.diag(torch.tensor([5e1, 5e1, 1e-1, 1e-1], device=device)))
        self.register_buffer("diag_P0", torch.diag(torch.tensor([1e-3, 1e-3, 1e-3, 1e-3, 1e-5], device=device))).unsqueeze(0)
        
    def _state_transition_func(self, x, ur):
        """
        x: [batch_size, sigma,  state_dim]
        """
        # ur = args[0].unsqueeze(1).repeat(1, x.shape[1], 1)
        xu = torch.cat((x, ur), dim=-1)
        x_next = ode.odeint(self.model, xu, self.dt, **self.solver_settngs)[-1]
        x_next = x_next[..., : self.state_dim]
        return x_next

    def _observation_func(self, x, ur):
        """
        x: [batch_size, sigma,  state_dim]
        """
        # ur = args[0].unsqueeze(1).repeat(1, x.shape[1], 1)
        xu = torch.cat((x, ur), dim=-1)
        
        
        
        
        
        return observation(self.model, xu)
    
    def forward(self, X_hat: Tensor, P: Tensor, u: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        X_hat, P = self.ukf.predict(X_hat, P, self.diag_Q, u)
        X_hat, P = self.ukf.update(X_hat, P, y, self.diag_R, u)
        return X_hat, P
