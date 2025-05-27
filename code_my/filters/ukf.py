import torch


class UKF:
    def __init__(self, state_dim: int, device, kappa=-2.0):
        self.state_dim = state_dim
        self.kappa = kappa

        self.weights_m = torch.zeros(2 * state_dim + 1, device=device)
        self.weights_c = torch.zeros(2 * state_dim + 1, device=device)

        self.weights_m[0] = self.kappa / (state_dim + self.kappa)
        self.weights_c[0] = self.kappa / (state_dim + self.kappa)

        self.weights_m[1:] = 1 / (2 * (state_dim + self.kappa))
        self.weights_c[1:] = 1 / (2 * (state_dim + self.kappa))

        self.device = device

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
    
    def make_positive_definite(self, matrix, eps=10):
        sym_matrix = 0.5 * (matrix + matrix.transpose(-2, -1))
        I= torch.eye(matrix.shape[-1], device=self.device)
        return sym_matrix + I * eps