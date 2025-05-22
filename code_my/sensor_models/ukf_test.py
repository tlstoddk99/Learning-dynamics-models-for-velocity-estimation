import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --------------------------
# Simulation & Model Settings
# --------------------------
dt = 0.1                # time step [s]
sim_time = 20.0         # total simulation time [s]
steps = int(sim_time / dt)
L = 2.5                 # vehicle wheelbase [m]
R_track = 20.0          # desired circular track radius [m]
v_nominal = 10.0        # nominal speed [m/s]

# Control input to follow a circle:
# For a kinematic bicycle model, the turning rate is (v/L)*tan(delta).
# To follow a circle of radius R_track at constant speed, we want v/R_track = (v/L)*tan(delta)
# => tan(delta) = L/R_track.
u_control = np.array([0.0, np.arctan(L / R_track)])  # [acceleration, steering]

# --------------------------
# Process & Measurement Noise Covariances
# --------------------------
# Process noise covariance (assumed small)
Q = np.diag([0.1, 0.1, 0.01, 0.1])
# Measurement noise covariance (we measure [x, y])
R = np.diag([1.0, 1.0])

# --------------------------
# UKF Parameters
# --------------------------
alpha = 1e-3   # primary scaling parameter (small positive value)
kappa = 0      # secondary scaling parameter (often set to 0)
beta = 2       # optimal for Gaussian distributions

# --------------------------
# Vehicle Dynamics (Process Model)
# --------------------------
def process_model(x, u, dt):
    """
    Kinematic bicycle model.
    x: state vector [x, y, theta, v]
    u: control input [acceleration, steering angle]
    """
    a, delta = u
    x_new = x[0] + x[3] * np.cos(x[2]) * dt
    y_new = x[1] + x[3] * np.sin(x[2]) * dt
    theta_new = x[2] + (x[3] / L) * np.tan(delta) * dt
    v_new = x[3] + a * dt
    return np.array([x_new, y_new, theta_new, v_new])

# --------------------------
# Measurement Model
# --------------------------
def measurement_model(x):
    """
    Returns the measurement vector.
    Here we assume we directly measure the vehicle position [x, y].
    """
    return np.array([x[0], x[1]])

# --------------------------
# Unscented Transform Helpers
# --------------------------
def generate_sigma_points(x, P, lambda_, n):
    """
    Generate 2n+1 sigma points for state x with covariance P.
    """
    sigma_points = np.zeros((2 * n + 1, n))
    sigma_points[0] = x
    sqrt_P = np.linalg.cholesky((n + lambda_) * P)
    for i in range(n):
        sigma_points[i + 1] = x + sqrt_P[:, i]
        sigma_points[n + i + 1] = x - sqrt_P[:, i]
    return sigma_points

# --------------------------
# UKF Prediction Step
# --------------------------
def ukf_predict(x, P, u, dt, Q, alpha, beta, kappa):
    n = len(x)
    lambda_ = alpha**2 * (n + kappa) - n
    # Generate sigma points
    sigma_points = generate_sigma_points(x, P, lambda_, n)
    # Propagate sigma points through the process model
    sigma_points_pred = np.array([process_model(sp, u, dt) for sp in sigma_points])
    # Compute weights
    Wm = np.full(2 * n + 1, 1.0 / (2 * (n + lambda_)))
    Wc = np.full(2 * n + 1, 1.0 / (2 * (n + lambda_)))
    Wm[0] = lambda_ / (n + lambda_)
    Wc[0] = lambda_ / (n + lambda_) + (1 - alpha**2 + beta)
    # Predicted state mean
    x_pred = np.sum(Wm[:, None] * sigma_points_pred, axis=0)
    # Predicted state covariance
    P_pred = Q.copy()
    for i in range(2 * n + 1):
        diff = sigma_points_pred[i] - x_pred
        P_pred += Wc[i] * np.outer(diff, diff)
    return x_pred, P_pred, sigma_points_pred, Wm, Wc

# --------------------------
# UKF Update Step
# --------------------------
def ukf_update(x_pred, P_pred, sigma_points_pred, Wm, Wc, z, R):
    n = len(x_pred)
    m = len(z)  # measurement dimension
    n_sigma = sigma_points_pred.shape[0]
    # Transform sigma points through the measurement model
    Zsig = np.array([measurement_model(sp) for sp in sigma_points_pred])
    # Predicted measurement mean
    z_pred = np.sum(Wm[:, None] * Zsig, axis=0)
    # Predicted measurement covariance
    S = R.copy()
    for i in range(n_sigma):
        diff_z = Zsig[i] - z_pred
        S += Wc[i] * np.outer(diff_z, diff_z)
    # Cross covariance between state and measurement
    Tc = np.zeros((n, m))
    for i in range(n_sigma):
        diff_x = sigma_points_pred[i] - x_pred
        diff_z = Zsig[i] - z_pred
        Tc += Wc[i] * np.outer(diff_x, diff_z)
    # Kalman gain
    K = Tc @ np.linalg.inv(S)
    # Update state estimate and covariance
    x_updated = x_pred + K @ (z - z_pred)
    P_updated = P_pred - K @ S @ K.T
    return x_updated, P_updated, z_pred, S, K

# --------------------------
# Simulation Initialization
# --------------------------
# True initial state (starting on the circle at (R_track,0), heading upward)
x_true = np.array([R_track, 0.0, np.pi/2, v_nominal])
# UKF initial estimate (with a small error)
x_est = x_true + np.array([0.5, -0.5, 0.1, 0.0])
P_est = np.diag([1.0, 1.0, 0.1, 1.0])

# Data history for visualization
true_history = [x_true.copy()]
est_history = [x_est.copy()]
meas_history = []  # measurements (position only)

# --------------------------
# Main Simulation Loop
# --------------------------
for _ in range(steps):
    # --- True State Propagation ---
    x_true = process_model(x_true, u_control, dt)
    true_history.append(x_true.copy())

    # --- Measurement Generation ---
    # Simulate a noisy measurement of [x, y]
    z = measurement_model(x_true) + np.random.multivariate_normal(np.zeros(2), R)
    meas_history.append(z)

    # --- UKF Prediction ---
    x_pred, P_pred, sigma_points_pred, Wm, Wc = ukf_predict(x_est, P_est, u_control, dt, Q, alpha, beta, kappa)
    # --- UKF Update ---
    x_est, P_est, z_pred, S, K = ukf_update(x_pred, P_pred, sigma_points_pred, Wm, Wc, z, R)
    est_history.append(x_est.copy())

# Convert histories to arrays for plotting
true_history = np.array(true_history)
est_history = np.array(est_history)
meas_history = np.array(meas_history)

# --------------------------
# Visualization with Matplotlib Animation
# --------------------------
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal')
ax.set_xlim(-R_track - 5, R_track + 5)
ax.set_ylim(-R_track - 5, R_track + 5)
ax.set_title("UKF State Estimation for Racing Vehicle")
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")

# Draw the circular track for reference
theta_track = np.linspace(0, 2 * np.pi, 200)
x_track = R_track * np.cos(theta_track)
y_track = R_track * np.sin(theta_track)
ax.plot(x_track, y_track, 'k--', label="Track")

# Plot elements to update during animation
true_line, = ax.plot([], [], 'b-', lw=2, label="True Trajectory")
est_line, = ax.plot([], [], 'g-', lw=2, label="UKF Estimate")
meas_scatter = ax.scatter([], [], c='r', marker='o', s=30, label="Measurements")
est_marker, = ax.plot([], [], 'go', markersize=8)
true_marker, = ax.plot([], [], 'bo', markersize=8)

ax.legend(loc='upper right')

def init():
    true_line.set_data([], [])
    est_line.set_data([], [])
    est_marker.set_data([], [])
    true_marker.set_data([], [])
    # Use an empty array with shape (0,2) instead of an empty list
    meas_scatter.set_offsets(np.empty((0, 2)))
    return true_line, est_line, est_marker, true_marker, meas_scatter

def animate(i):
    # Update true and estimated trajectory lines
    true_line.set_data(true_history[:i+1, 0], true_history[:i+1, 1])
    est_line.set_data(est_history[:i+1, 0], est_history[:i+1, 1])
    
    # Update markers (wrap scalars in lists)
    true_marker.set_data([true_history[i, 0]], [true_history[i, 1]])
    est_marker.set_data([est_history[i, 0]], [est_history[i, 1]])
    
    # For measurements, show all measurements up to current step
    if i > 0:
        meas_scatter.set_offsets(meas_history[:i])
    else:
        meas_scatter.set_offsets(np.empty((0, 2)))
    return true_line, est_line, true_marker, est_marker, meas_scatter


# If you still encounter issues with '_resize_id', try removing blit=True:
ani = animation.FuncAnimation(fig, animate, frames=len(true_history),
                              init_func=init, interval=50, blit=True)

plt.show()
