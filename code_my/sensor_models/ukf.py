import torch
from torch import Tensor
import torch.nn as nn
from typing import Tuple, Callable

class UKF(nn.Module):
    state_dim: int
    meas_dim: int
    kappa: float
    weights_m: Tensor
    weights_c: Tensor

    def __init__(
        self,
        state_dim: int,
        meas_dim: int,
        device: torch.device,
        kappa: float = -2.0
    ) -> None:
        super(UKF, self).__init__()
        self.state_dim = state_dim
        self.meas_dim = meas_dim
        self.kappa = kappa

        denom = state_dim + kappa
        wm0 = kappa / denom
        wc0 = wm0
        w = 1.0 / (2.0 * denom)

        weights_m = torch.full((2 * state_dim + 1,), w, device=device)
        weights_c = torch.full((2 * state_dim + 1,), w, device=device)
        weights_m[0] = wm0
        weights_c[0] = wc0

        # Register as buffers so they're moved with the module/device
        self.register_buffer('weights_m', weights_m)
        self.register_buffer('weights_c', weights_c)

    def predict(
        self,
        X: Tensor,
        P: Tensor,
        Q: Tensor,
        state_fn: Callable[[Tensor], Tensor]
    ) -> Tuple[Tensor, Tensor]:
        sigma = self.generate_sigma_points(X, P)
        sigma_prop = state_fn(sigma)
        X_pred, P_pred = self.recover_gaussian(sigma_prop)
        P_pred = P_pred + Q
        return X_pred, P_pred

    def update(
        self,
        X_pred: Tensor,
        P_pred: Tensor,
        Z: Tensor,
        R: Tensor,
        meas_fn: Callable[[Tensor], Tensor]
    ) -> Tuple[Tensor, Tensor]:
        sigma = self.generate_sigma_points(X_pred, P_pred)
        meas_sigma = meas_fn(sigma)
        z_pred, Pz = self.recover_gaussian(meas_sigma)
        Pxz = self.cross_covariance(sigma, meas_sigma, X_pred, z_pred)

        # Compute Kalman gain
        M = Pz + R
        M_inv = torch.inverse(M)
        K = Pxz @ M_inv

        # Update state
        y = Z - z_pred
        X_upd = X_pred + (K @ y.unsqueeze(-1)).squeeze(-1)
        P_upd = P_pred - K @ Pz @ K.transpose(-2, -1)
        return X_upd, P_upd

    def generate_sigma_points(
        self,
        X: Tensor,
        P: Tensor
    ) -> Tensor:
        # X: [batch, state_dim], P: [batch, state_dim, state_dim]
        scale = self.state_dim + self.kappa
        sqrtm = torch.linalg.cholesky(scale * P)
        X_exp = X.unsqueeze(1)
        pos = sqrtm.transpose(-1, -2)
        sig_pos = X_exp + pos
        sig_neg = X_exp - pos
        sigma = torch.cat([X_exp, sig_pos, sig_neg], dim=1)
        return sigma

    def recover_gaussian(
        self,
        sigma: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # sigma: [batch, 2*state_dim+1, dim]
        mean = torch.einsum('i,bij->bj', self.weights_m, sigma)
        diff = sigma - mean.unsqueeze(1)
        cov = torch.einsum('i,bij,bik->bjk', self.weights_c, diff, diff)
        return mean, cov

    def cross_covariance(
        self,
        X_sigma: Tensor,
        Z_sigma: Tensor,
        X_mean: Tensor,
        Z_mean: Tensor
    ) -> Tensor:
        diff_x = X_sigma - X_mean.unsqueeze(1)
        diff_z = Z_sigma - Z_mean.unsqueeze(1)
        Pxz = torch.einsum('i,bij,bik->bjk', self.weights_c, diff_x, diff_z)
        return Pxz





if __name__ == '__main__':

    def state_fn(sigma: torch.Tensor) -> torch.Tensor:
        return sigma

    def meas_fn(sigma: torch.Tensor) -> torch.Tensor:
        return sigma

    def ukf_model() -> UKF:
        device = torch.device('cpu')
        return UKF(state_dim=3, meas_dim=3, device=device)

    def test_predict_identity(ukf_model: UKF):
        batch = 2
        X = torch.randn(batch, 3)
        P = torch.eye(3).unsqueeze(0).repeat(batch, 1, 1)
        Q = torch.zeros(batch, 3, 3)
        X_pred, P_pred = ukf_model.predict(X, P, Q, state_fn)
        # The state and covariance should be unchanged for identity transform
        assert torch.allclose(X_pred, X, atol=1e-6)
        assert torch.allclose(P_pred, P, atol=1e-6)

    def test_update_identity(ukf_model: UKF):
        batch = 2
        X_pred = torch.randn(batch, 3)
        P_pred = torch.eye(3).unsqueeze(0).repeat(batch, 1, 1)
        Z = X_pred.clone()
        R = torch.zeros(batch, 3, 3)
        X_upd, P_upd = ukf_model.update(X_pred, P_pred, Z, R, meas_fn)
        # With perfect measurements, the state should remain the same
        assert torch.allclose(X_upd, X_pred, atol=1e-6)
        assert torch.allclose(P_upd, P_pred, atol=1e-6)

    def test_scriptable():
        # Ensure the module can be scripted without errors
        device = torch.device('cpu')
        ukf = UKF(state_dim=2, meas_dim=2, device=device)
        scripted = torch.jit.script(ukf)
        assert isinstance(scripted, torch.jit.ScriptModule)