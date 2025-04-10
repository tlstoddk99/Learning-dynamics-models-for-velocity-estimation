import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math
import os

#############################################
# 1. Vehicle Constants and Parameters
#############################################
# Example physical constants for a 1/10-scale racing car.
m = 4.5          # mass [kg]
Iz = 0.5         # moment of inertia about z [kg·m²]
lf = 0.15        # distance from the center-of-mass to front axle [m]
lr = 0.15        # distance from the center-of-mass to rear axle [m]
rho = 0.03       # effective wheel radius [m]
Ie = 0.001       # motor inertia [kg·m²]
k_phi = 1.0      # motor torque constant
c_d = 0.1        # aerodynamic drag coefficient (F_drag = c_d * vx**2)

# Transmission friction parameters
k_tC = 0.01      # Coulomb friction coefficient
k_tv = 0.001     # Viscous friction coefficient

#############################################
# 2. Neural Tire Force Model (Friction–scaled)
#############################################
class TireForceNN(nn.Module):
    """
    Neural network to predict tire forces.
    Input: concatenation of state and control input.
    Output: a 4D vector [Fxf, Fxr, Fyf, Fyr] representing raw tire forces.
    The predicted forces will be later scaled by the friction coefficient μ.
    """
    def __init__(self, input_dim, hidden_dim=64, output_dim=4):
        super(TireForceNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, state, u):
        # Concatenate the state and control inputs along the last dimension.
        inp = torch.cat([state, u], dim=-1)
        return self.net(inp)

#############################################
# 3. Vehicle Dynamics Model (NNTF Model)
#############################################
class VehicleDynamics(nn.Module):
    """
    Vehicle dynamics using a single–track model with a neural network-based tire model.
    Tire forces are computed by applying a neural network whose output is scaled by the 
    current friction coefficient μ (learned online via UKF).
    
    State x: [vx, vy, r, ωs, μ]
       vx, vy: vehicle longitudinal and lateral velocities,
       r: yaw rate,
       ωs: wheel speed,
       μ: tire–road friction coefficient.
    Control u: [δ, Iq]
       δ: steering angle,
       Iq: motor current.
    """
    def __init__(self):
        super(VehicleDynamics, self).__init__()
        # Input dimension: state (5) + control (2)
        self.tire_nn = TireForceNN(input_dim=7, hidden_dim=64, output_dim=4)

    def forward(self, x, u):
        # Unpack state components: x shape is (batch, 5)
        vx, vy, r, ws, mu = torch.chunk(x, chunks=5, dim=-1)
        
        # Predict raw tire forces and scale them by friction coefficient (NNTF model)
        tire_forces_raw = self.tire_nn(x, u)  # shape (batch, 4)
        tire_forces = mu * tire_forces_raw   # scale by μ
        
        # Unpack tire forces
        Fxf = tire_forces[:, 0:1]
        Fxr = tire_forces[:, 1:2]
        Fyf = tire_forces[:, 2:3]
        Fyr = tire_forces[:, 3:4]
        
        # Aerodynamic drag force (quadratic in vx)
        F_drag = c_d * vx**2
        
        # Unpack control inputs: steering angle δ and motor current Iq
        delta, Iq = torch.chunk(u, chunks=2, dim=-1)
        cos_delta = torch.cos(delta)
        sin_delta = torch.sin(delta)
        
        # Dynamics equations (as in Equation (2) of the paper):
        vx_dot = (1.0/m) * (Fxr + Fxf * cos_delta - Fyf * sin_delta - F_drag + m * vy * r)
        vy_dot = (1.0/m) * (Fxf * sin_delta + Fyr + Fyf * cos_delta - m * vx * r)
        r_dot = (1.0/Iz) * (((Fxf * sin_delta + Fyf * cos_delta) * lf) - Fyr * lr)
        tau_t = k_tC * torch.sign(ws) + k_tv * ws  # transmission friction torque
        ws_dot = (1.0/Ie) * (k_phi * Iq - rho * (Fxf + Fxr) - tau_t)
        # Friction coefficient assumed constant during dynamics propagation
        mu_dot = torch.zeros_like(mu)
        
        # Concatenate state derivatives into one vector
        dxdt = torch.cat([vx_dot, vy_dot, r_dot, ws_dot, mu_dot], dim=-1)
        return dxdt

#############################################
# 4. RK4 Integration
#############################################
def rk4_step(model, x, u, dt):
    """
    A single integration step using the 4th order Runge-Kutta method.
    """
    k1 = model(x, u)
    k2 = model(x + dt/2.0 * k1, u)
    k3 = model(x + dt/2.0 * k2, u)
    k4 = model(x + dt * k3, u)
    x_next = x + dt/6.0 * (k1 + 2*k2 + 2*k3 + k4)
    return x_next

#############################################
# 5. Measurement Function
#############################################
def measurement_function(x, u, dynamics_model, dt):
    """
    Predicts the measurement vector based on the state x.
    Measurements include: yaw rate (r), wheel speed (ws), and inertial accelerations a_x and a_y.
    
    The IMU measures:
       a_x = vx_dot - r*vy,
       a_y = vy_dot + r*vx.
    """
    dxdt = dynamics_model(x, u)
    vx_dot = dxdt[:, 0:1]
    vy_dot = dxdt[:, 1:2]
    vx, vy, r, ws, _ = torch.chunk(x, chunks=5, dim=-1)
    a_x = vx_dot - r * vy
    a_y = vy_dot + r * vx
    # Measurement vector: [r, ws, a_x, a_y]
    y_pred = torch.cat([r, ws, a_x, a_y], dim=-1)
    return y_pred

def nearest_pd(P, eps=1e-6):
    """
    Returns the nearest positive-definite matrix to the input.
    
    Args:
        P (Tensor): A symmetric matrix of shape (batch, n, n).
        eps (float): A small value to ensure eigenvalues are strictly positive.
        
    Returns:
        Tensor: The nearest positive-definite matrix.
    """
    # Ensure symmetry
    P_sym = 0.5 * (P + P.transpose(-1, -2))
    # Compute eigen-decomposition
    eigvals, eigvecs = torch.linalg.eigh(P_sym)
    # Clamp eigenvalues to be at least eps
    eigvals_clamped = torch.clamp(eigvals, min=eps)
    # Reconstruct the matrix
    P_pd = (eigvecs * eigvals_clamped.unsqueeze(-2)) @ eigvecs.transpose(-1, -2)
    return P_pd


#############################################
# 6. UKF Sigma-Point Generation with Jitter
#############################################
def generate_sigma_points(x, P, alpha, beta, kappa):
    """
    Generate sigma points and associated weights for the Unscented Kalman Filter (UKF).

    This function creates 2n+1 sigma points from a state vector x and covariance matrix P,
    using the Unscented Transform. If the covariance is not positive definite (PD),
    it increases a jitter value, and if necessary, applies a nearest positive-definite
    adjustment until the Cholesky factorization succeeds.

    Args:
        x (Tensor): The state mean vector of shape (batch, n).
        P (Tensor): The state covariance matrix of shape (batch, n, n).
        alpha (float): Spread scaling parameter (usually a small positive value, e.g., 1e-3).
        beta (float): Prior knowledge parameter (for Gaussian distributions, beta=2 is often optimal).
        kappa (float): Secondary scaling parameter.

    Returns:
        sigma_points (Tensor): Sigma points of shape (batch, 2n+1, n).
        Wm (Tensor): Weights for the mean, shape (2n+1,).
        Wc (Tensor): Weights for the covariance, shape (2n+1,).
    """
    n = x.shape[-1]
    lambda_ = alpha**2 * (n + kappa) - n

    # Force symmetry on covariance matrix for numerical stability.
    P = 0.5 * (P + P.transpose(-1, -2))

    # Prepare to construct the scaled covariance needed for sigma point generation.
    base_matrix = (n + lambda_) * P
    I = torch.eye(n, device=P.device).unsqueeze(0).expand_as(P)
    
    # Initialize jitter.
    jitter = 1e-5
    max_iter = 10
    for i in range(max_iter):
        try:
            # Try Cholesky decomposition on the modified matrix.
            sqrt_P = torch.linalg.cholesky(base_matrix + jitter * I)
            break  # Success: exit the loop.
        except RuntimeError as e:
            # Every few iterations, attempt to adjust base_matrix to the nearest PD matrix.
            if i % 3 == 2:
                base_matrix = nearest_pd(base_matrix)
            jitter *= 10  # Increase the jitter multiplier.
            if i == max_iter - 1:
                raise RuntimeError("Covariance not PD even after adding jitter and nearest PD adjustment") from e

    # Generate sigma points.
    sigma_points = [x]  # First sigma point is the mean.
    for i in range(n):
        sigma_points.append(x + sqrt_P[:, :, i])
        sigma_points.append(x - sqrt_P[:, :, i])
    sigma_points = torch.stack(sigma_points, dim=1)  # Shape: (batch, 2n+1, n)

    # Compute weights for the sigma points.
    weight = 1.0 / (2 * (n + lambda_))
    Wm = torch.full((2 * n + 1,), weight, device=x.device)
    Wc = torch.full((2 * n + 1,), weight, device=x.device)
    Wm[0] = lambda_ / (n + lambda_)
    Wc[0] = lambda_ / (n + lambda_) + (1 - alpha**2 + beta)

    return sigma_points, Wm, Wc

#############################################
# 7. Differentiable UKF Implementation with Learnable Noise Models
#############################################
class DifferentiableUKF(nn.Module):
    """
    Differentiable Unscented Kalman Filter.
    Propagates sigma points through the vehicle dynamics (via RK4) and updates
    with a measurement function.
    
    Noise covariances are parameterized using lower–triangular matrices LQ and LR:
       Q = LQ @ LQᵀ + ε I,  R = LR @ LRᵀ + ε I,
    which enforces positive definiteness.
    """
    def __init__(self, state_dim, meas_dim, dynamics_model, meas_function,
                 dt, alpha=0.5, beta=2.0, kappa=0.0):
        super(DifferentiableUKF, self).__init__()
        self.state_dim = state_dim
        self.meas_dim = meas_dim
        self.dynamics_model = dynamics_model
        self.meas_function = meas_function
        self.dt = dt
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        # Learnable lower triangular matrices for process and measurement noise
        self.LQ = nn.Parameter(torch.eye(state_dim))
        self.LR = nn.Parameter(torch.eye(meas_dim))
        self.noise_epsilon = 1e-7  # small constant for numerical stability

    def get_process_noise(self):
        # Q = LQ * LQ.T + εI
        LQ = torch.tril(self.LQ)  # enforce lower triangular structure
        Q = LQ @ LQ.transpose(0, 1) + self.noise_epsilon * torch.eye(self.state_dim, device=LQ.device)
        return Q

    def get_measurement_noise(self):
        # R = LR * LR.T + εI
        LR = torch.tril(self.LR)  # enforce lower triangular structure
        R = LR @ LR.transpose(0, 1) + self.noise_epsilon * torch.eye(self.meas_dim, device=LR.device)
        return R

    def forward(self, x, P, u, y):
        """
        One UKF update step.
          x: current state estimate (batch, state_dim)
          P: current covariance (batch, state_dim, state_dim)
          u: control input (batch, control_dim)
          y: measurement (batch, meas_dim)
        Returns: updated state and covariance along with intermediates.
        """
        batch_size = x.shape[0]
        n = self.state_dim
        m = self.meas_dim

        # Generate sigma points and weights
        sigma_points, Wm, Wc = generate_sigma_points(x, P, self.alpha, self.beta, self.kappa)
        # Propagate sigma points through the dynamics using RK4 integration
        sigma_points_pred = []
        for i in range(sigma_points.shape[1]):
            sp = sigma_points[:, i, :]  # shape: (batch, n)
            sp_pred = rk4_step(self.dynamics_model, sp, u, self.dt)
            sigma_points_pred.append(sp_pred)
        sigma_points_pred = torch.stack(sigma_points_pred, dim=1)  # (batch, 2n+1, n)
        
        # Predicted state mean (weighted sum)
        Wm_tensor = Wm.view(1, -1, 1)
        x_pred = torch.sum(Wm_tensor * sigma_points_pred, dim=1)  # (batch, n)
        
        # Predicted state covariance
        diff_x = sigma_points_pred - x_pred.unsqueeze(1)  # (batch, 2n+1, n)
        Wc_tensor = Wc.view(1, -1, 1, 1)
        P_pred = torch.sum(Wc_tensor * (diff_x.unsqueeze(-1) @ diff_x.unsqueeze(-2)), dim=1) + self.get_process_noise()

        # Generate predicted measurements from each sigma point
        sigma_meas = []
        for i in range(sigma_points_pred.shape[1]):
            sp = sigma_points_pred[:, i, :]  # (batch, n)
            meas_pred = self.meas_function(sp, u, self.dynamics_model, self.dt)  # (batch, m)
            sigma_meas.append(meas_pred)
        sigma_meas = torch.stack(sigma_meas, dim=1)  # (batch, 2n+1, m)

        # Predicted measurement mean
        y_pred = torch.sum(Wm_tensor * sigma_meas, dim=1)  # (batch, m)

        # Measurement covariance and cross-covariance
        diff_y = sigma_meas - y_pred.unsqueeze(1)  # (batch, 2n+1, m)
        S = torch.sum(Wc_tensor * (diff_y.unsqueeze(-1) @ diff_y.unsqueeze(-2)), dim=1) + self.get_measurement_noise()
        cross_cov = torch.sum(Wc_tensor * (diff_x.unsqueeze(-1) @ diff_y.unsqueeze(-2)), dim=1)

        # Regularize S before inversion
        epsilon = 1e-5
        S_inv = torch.linalg.inv(S + epsilon * torch.eye(m, device=S.device))
        K = torch.matmul(cross_cov, S_inv)  # Kalman gain (batch, n, m)

        # Innovation and update
        innovation = y - y_pred  # (batch, m)
        x_updated = x_pred + (K @ innovation.unsqueeze(-1)).squeeze(-1)
        P_updated = P_pred - K @ S @ K.transpose(1, 2)

        # Enforce symmetry on the updated covariance:
        P_updated = 0.5 * (P_updated + P_updated.transpose(1, 2))
        # Add a diagonal jitter to ensure positive definiteness:
        P_updated = P_updated + 1e-4 * torch.eye(self.state_dim, device=P_updated.device).unsqueeze(0)

        return x_updated, P_updated, x_pred, P_pred, y_pred, S, K

#############################################
# 8. Pretraining Dynamics Model via One-Step Prediction Loss
#############################################
def pretrain_dynamics(dynamics_model, data_loader, num_epochs=100, lr=5e-4, dt=0.01, device="cpu"):
    """
    Pretrain the dynamics model to be a good one-step predictor.
    This stage minimizes the one-step prediction error:
       L = || x_pred - x_true_next ||²,
    where x_pred is computed by integrating the dynamics via RK4.
    """
    dynamics_model.to(device)
    optimizer = optim.Adam(dynamics_model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        count = 0
        for (x0, _, u_seq, _, x_true_seq) in data_loader:
            # x_true_seq: shape (batch, T, state_dim)
            # We'll use steps from t=0 to T-1 for one-step prediction.
            x0 = x0.to(device)
            u_seq = u_seq.to(device)
            x_true_seq = x_true_seq.to(device)
            batch_size, T, _ = u_seq.shape

            loss_batch = 0.0
            x_current = x0  # initial state
            # Predict one step at a time for each sequence
            for t in range(T - 1):
                u_t = u_seq[:, t, :]
                x_true_next = x_true_seq[:, t+1, :]
                x_pred = rk4_step(dynamics_model, x_current, u_t, dt)
                loss_batch += mse_loss(x_pred, x_true_next)
                # For pretraining, we feed the ground truth for the next step.
                x_current = x_true_seq[:, t+1, :]
            loss_batch = loss_batch / (T - 1)
            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()
            epoch_loss += loss_batch.item()
            count += 1

        print(f"[Pretraining] Epoch {epoch+1}/{num_epochs} Loss: {epoch_loss/count:.6f}")

#############################################
# 9. Training Loop for Differentiable UKF (End-to-End Fine-Tuning)
#############################################
def train_ukf(model_ukf, dynamics_model, data_loader, num_epochs=20, lr=5e-4, device="cpu"):
    """
    End-to-end training loop where the UKF (and implicitly the dynamics model)
    is optimized to minimize the state estimation error over a sequence.
    """
    optimizer = optim.Adam(model_ukf.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    model_ukf.to(device)
    dynamics_model.to(device)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for (x0, P0, u_seq, y_seq, x_true_seq) in data_loader:
            # Move tensors to device
            x0 = x0.to(device)
            P0 = P0.to(device)
            u_seq = u_seq.to(device)         # shape: (batch, T, control_dim)
            y_seq = y_seq.to(device)         # shape: (batch, T, meas_dim)
            x_true_seq = x_true_seq.to(device)  # shape: (batch, T, state_dim)
            
            batch_size, T, _ = u_seq.shape

            # Initialize state and covariance
            x_est = x0  # (batch, state_dim)
            P_est = P0  # (batch, state_dim, state_dim)
            loss_seq = 0.0

            # Roll through the sequence
            for t in range(T):
                u_t = u_seq[:, t, :]
                y_t = y_seq[:, t, :]
                x_est, P_est, _, _, y_pred, _, _ = model_ukf(x_est, P_est, u_t, y_t)
                # Accumulate loss over the sequence (MSE against ground truth state)
                loss_seq = loss_seq + mse_loss(x_est, x_true_seq[:, t, :])
            loss_seq = loss_seq / T

            optimizer.zero_grad()
            loss_seq.backward()
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model_ukf.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss_seq.item()

        print(f"[UKF Fine-Tuning] Epoch {epoch+1}/{num_epochs} Loss: {epoch_loss/len(data_loader):.6f}")

#############################################
# 10. Synthetic Dataset for Demonstration
#############################################
def create_synthetic_dataset(num_samples=200, seq_len=50, state_dim=5, control_dim=2, meas_dim=4):
    """
    Creates a synthetic dataset for demonstration purposes.
    In a real application, replace this with your preprocessed data.
    """
    x0 = torch.zeros(num_samples, state_dim)  # initial state (zeros)
    P0 = torch.stack([torch.eye(state_dim) for _ in range(num_samples)])  # identity covariance for all samples
    # Create random control inputs, measurements, and true state sequences.
    u_seq = torch.randn(num_samples, seq_len, control_dim)
    x_true_seq = torch.randn(num_samples, seq_len, state_dim)
    y_seq = torch.randn(num_samples, seq_len, meas_dim)
    
    dataset = TensorDataset(x0, P0, u_seq, y_seq, x_true_seq)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    return loader

#############################################
# 11. Model Saving Functionality
#############################################
def save_models(dynamics_model, ukf_model, dynamics_path, ukf_path):
    torch.save(dynamics_model.state_dict(), os.path.join(dynamics_path, "dynamics_model.pt"))
    torch.save(ukf_model.state_dict(), os.path.join(ukf_path, "ukf_model.pt"))
    print("Models saved successfully.")

#############################################
# 12. Testing the Trained Model
#############################################
def test_ukf(model_ukf, dynamics_model, data_loader, device="cpu"):
    """
    Test the trained UKF by computing the average state estimation MSE on the test dataset.
    """
    mse_loss = nn.MSELoss()
    total_loss = 0.0
    count = 0
    model_ukf.eval()
    dynamics_model.eval()
    with torch.no_grad():
        for (x0, P0, u_seq, y_seq, x_true_seq) in data_loader:
            x0 = x0.to(device)
            P0 = P0.to(device)
            u_seq = u_seq.to(device)
            y_seq = y_seq.to(device)
            x_true_seq = x_true_seq.to(device)
            batch_size, T, _ = u_seq.shape
            x_est = x0
            P_est = P0
            loss_seq = 0.0
            for t in range(T):
                u_t = u_seq[:, t, :]
                y_t = y_seq[:, t, :]
                x_est, P_est, _, _, y_pred, _, _ = model_ukf(x_est, P_est, u_t, y_t)
                loss_seq += mse_loss(x_est, x_true_seq[:, t, :])
            total_loss += loss_seq.item() / T
            count += 1
    avg_loss = total_loss / count
    print("Test Average MSE Loss:", avg_loss)

#############################################
# 13. Main Script: Pretraining, UKF Fine-Tuning, Saving and Testing
#############################################
if __name__ == "__main__":
    output_path = "/home/a/Learning-dynamics-models-for-velocity-estimation/model_test/output"
    dynamics_save_path = os.path.join(output_path, "dynamics_model")
    ukf_save_path = os.path.join(output_path, "ukf_model")
    
    # Create directories if they don't exist
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(ukf_save_path, exist_ok=True)
    os.makedirs(dynamics_save_path, exist_ok=True)
    
    # Select device (GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Define dimensions
    state_dim = 5   # [vx, vy, r, ws, μ]
    control_dim = 2 # [δ, Iq]
    meas_dim = 4    # [r, ws, a_x, a_y]
    dt = 0.01       # time step in seconds
    
    # Initialize dynamics and UKF models
    dynamics_model = VehicleDynamics()
    # Using UKF parameters alpha=0.5, beta=2.0, kappa=0.0 as in the paper
    ukf_filter = DifferentiableUKF(state_dim, meas_dim, dynamics_model, measurement_function, dt,
                                   alpha=0.5, beta=2.0, kappa=0.0)
    
    # Create a synthetic dataset for training (replace with your real data)
    train_loader = create_synthetic_dataset(num_samples=200, seq_len=50,
                                            state_dim=state_dim, control_dim=control_dim, meas_dim=meas_dim)
    
    # --------- Stage 1: Pretrain Dynamics Model with One-Step Prediction Loss ----------
    print("Starting pretraining of the dynamics model...")
    pretrain_dynamics(dynamics_model, train_loader, num_epochs=100, lr=5e-4, dt=dt, device=device)
    torch.save(dynamics_model.state_dict(), os.path.join(dynamics_save_path, "dynamics_model_pretrained.pt"))
    print("Dynamics model pretraining completed.")
    
    # --------- Stage 2: Fine-Tune with End-to-End Differentiable UKF ----------
    torch.load_state_dict(dynamics_model, os.path.join(dynamics_save_path, "dynamics_model_pretrained.pt"))
    dynamics_model.to(device)
    ukf_filter.to(device)
    print("Starting UKF fine-tuning...")
    train_ukf(ukf_filter, dynamics_model, train_loader, num_epochs=20, lr=5e-4, device=device)
    
    # --------- Saving Models ----------
    save_models(dynamics_model, ukf_filter, dynamics_save_path, ukf_save_path)
    
    # --------- Testing the Trained Model ----------
    # Create a synthetic test dataset (replace with your real test data)
    test_loader = create_synthetic_dataset(num_samples=50, seq_len=50,
                                           state_dim=state_dim, control_dim=control_dim, meas_dim=meas_dim)
    print("Testing the trained UKF model...")
    test_ukf(ukf_filter, dynamics_model, test_loader, device=device)
