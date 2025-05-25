import torch


class UKF:
    def __init__(self, state_dim: int, meas_dim: int, device, kappa=-2.0):
        self.state_dim = state_dim
        self.meas_dim = meas_dim
        self.kappa = kappa

        self.weights_m = torch.zeros(2 * state_dim + 1, device=device)
        self.weights_c = torch.zeros(2 * state_dim + 1, device=device)

        self.weights_m[0] = self.kappa / (state_dim + self.kappa)
        self.weights_c[0] = self.kappa / (state_dim + self.kappa)

        self.weights_m[1:] = 1 / (2 * (state_dim + self.kappa))
        self.weights_c[1:] = 1 / (2 * (state_dim + self.kappa))

        self.device = device

    def predict(self, X, P, Q, state_transition_func, *args):
        """
        Predict the next state.
        X: State estimate [batch, state_dim]
        P: State covariance [batch, state_dim, state_dim]
        state_transition_func: Function to propagate state
        """
        assert X.shape[-1] == self.state_dim
        assert P.shape[-1] == self.state_dim

        sigma_points = self._generate_sigma_points(X, P)
        sigma_points_prop = state_transition_func(sigma_points, *args)
        assert sigma_points_prop.shape == sigma_points.shape

        X_pred, P_pred = self._recover_gaussian(sigma_points_prop)
        P_pred = P_pred + Q
        return X_pred, P_pred

    def update(self, X_pred, P_pred, Z, R, measurement_func, *args):
        """
        Update state estimate and covariance.
        X_pred: Predicted state estimate [batch, state_dim]
        P_pred: Predicted state covariance [batch, state_dim, state_dim]
        Z: Measurement [batch, meas_dim]
        measurement_func: Function to convert state to measurement
        """
        assert X_pred.shape[-1] == self.state_dim
        assert P_pred.shape[-1] == self.state_dim
        assert Z.shape[-1] == self.meas_dim

        sigma_points = self._generate_sigma_points(X_pred, P_pred)

        sigma_points_meas = measurement_func(sigma_points, *args)

        z_pred, Pz = self._recover_gaussian(sigma_points_meas)
        Pxz = self._cross_covariance(
            sigma_points, sigma_points_meas, X_pred, z_pred)

        inv_Pz_u = torch.cholesky_inverse(torch.linalg.cholesky(Pz + R))

        K = torch.matmul(Pxz, inv_Pz_u)

        X_updated = X_pred + \
            torch.matmul(K, (Z - z_pred).unsqueeze(-1)).squeeze(-1)
        P_updated = P_pred - \
            torch.matmul(K, torch.matmul(Pz, K.transpose(-2, -1)))

        return X_updated, P_updated

    def _generate_sigma_points(self, X, P):

        X_expanded = X.unsqueeze(1)
        sqrt_matrix = torch.linalg.cholesky((self.state_dim + self.kappa) * P)
        sigma_points_pos = X_expanded + sqrt_matrix.transpose(-1, -2)
        sigma_points_neg = X_expanded - sqrt_matrix.transpose(-1, -2)

        sigma_points = torch.cat(
            [X_expanded, sigma_points_pos, sigma_points_neg], dim=1)

        sigma_points = sigma_points.reshape(
            X.shape[0], 2 * self.state_dim + 1, self.state_dim)

        return sigma_points

    def _recover_gaussian(self, sigma_points):
        mean = torch.einsum('i,bin->bn', self.weights_m, sigma_points)
        diff = sigma_points - mean.unsqueeze(1)
        covariance = torch.einsum('i,bin,bis->bns', self.weights_c, diff, diff)
        return mean.squeeze(1), covariance

    def _cross_covariance(self, X_sigma, Z_sigma, X_mean, Z_mean):
        diff_x = X_sigma - X_mean.unsqueeze(-2)
        diff_z = Z_sigma - Z_mean.unsqueeze(-2)
        Pxz = torch.einsum('i,bix,biz->bxz', self.weights_c, diff_x, diff_z)
        return Pxz
