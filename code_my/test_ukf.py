import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import torch
import pickle
from sklearn.preprocessing import StandardScaler
import torchdiffeq as ode
from torch.utils.data import DataLoader
from filters.ukf_dataset import UkfDataset
from filters.ukf import UKF
from robot_models.single_track_pacejka import SingleTrackPacejkaModel
from robot_models.pacejka import PacejkaTireModel
from robot_models.single_track_parameters import SingleTrackParameters
from sensor_models.imu_model import ImuModel


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths and logging setup
df = pd.read_csv('/home/a/Learning-dynamics-models-for-velocity-estimation/code_my/dataset/hoons_all_test_gt.csv', index_col=0)


# Load the model state dict
sensor_model_path = torch.load(
    '/home/a/Learning-dynamics-models-for-velocity-estimation/code_my/trained_models/05-26_22-48/best_epoch_199_loss_1.5399.pt',
                              )
dataset= UkfDataset(df,run_id_list=[21])
test_dataloader= DataLoader(dataset, batch_size=1, shuffle=False)
#  ['v_x', 'v_y', 'r', 'omega_wheels', 'friction', 'delta', 'Iq', 'ax_imu', 'ay_imu', 'r_imu']
timestamp = time.strftime('%m-%d_%H-%M')

# fix seed
torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

vehicle_params = SingleTrackParameters()
tire_model = PacejkaTireModel(vehicle_parameters=vehicle_params)
vehicle_model = SingleTrackPacejkaModel(
    vehicle_parameters=vehicle_params,
    tire_model=tire_model
)
vehicle_model.to(device)
vehicle_model = torch.jit.script(vehicle_model)

sensor_model = ImuModel()
sensor_model.load_state_dict(sensor_model_path)
sensor_model.to(device)
sensor_model = torch.jit.script(sensor_model)

ukf = UKF(state_dim=5, device=device)

def observation(model, x, u):
    u_expanded = u[:, None, :].expand(-1, x.shape[1], -1)
    xu= torch.cat([x, u_expanded], dim=-1)  # (B, 11, 7)
    x_dot = model.forward(torch.tensor(0.0), xu) # (B, 11, 7)
    x_dot = x_dot[:,:, :5]  # (B, 11, 5)
    #x.shape: (B,5), 
    v_x, v_y, r, omega_wheels, friction = torch.unbind(x, dim=-1)
    v_x_dot, v_y_dot, r_dot, omega_wheels_dot, friction_dot = torch.unbind(x_dot, dim=-1)
    a_x = v_x_dot - r * v_y
    a_y = v_y_dot + r * v_x
    return torch.stack([a_x, a_y, r, omega_wheels], dim=-1)

def predict(X, P, Q, u):
    """
    Predict the next state.
    X: State estimate [batch, state_dim]
    P: State covariance [batch, state_dim, state_dim]
    state_transition_func: Function to propagate state
    """
    assert X.shape[-1] == 5
    assert P.shape[-1] == 5

    sigma_points = ukf._generate_sigma_points(X, P) # (B, 11, 5)
    u_expanded = u[:, None, :].expand(-1, sigma_points.shape[1], -1)
    xu= torch.cat([sigma_points, u_expanded], dim=-1)  # (B, 11, 7)

    sigma_points_prop = ode.odeint(
        vehicle_model,
        xu,
        torch.tensor([0, 0.01], device=sigma_points.device),
        method="rk4",
        rtol=1e-5,
        atol=1e-6
    )[-1] # (B, 11, 7)
    sigma_points_prop = sigma_points_prop[..., :5]  # (B, 11, 5)
    

    assert sigma_points_prop.shape == sigma_points.shape

    X_pred, P_pred = ukf._recover_gaussian(sigma_points_prop)
    P_pred = P_pred + Q
    return X_pred, P_pred

def update(X_pred, P_pred, Z, R, u):
    """
    Update state estimate and covariance.
    X_pred: Predicted state estimate [batch, state_dim]
    P_pred: Predicted state covariance [batch, state_dim, state_dim]
    Z: Measurement [batch, meas_dim]
    measurement_func: Function to convert state to measurement
    """
    assert X_pred.shape[-1] == 5
    assert P_pred.shape[-1] == 5
    assert Z.shape[-1] == 4

    sigma_points = ukf._generate_sigma_points(X_pred, P_pred) # (B, 11, 5)

    # sigma_points_meas = measurement_func(sigma_points)
    sigma_points_meas = observation(vehicle_model, sigma_points, u) # (B, 11, 4)

    z_pred, Pz = ukf._recover_gaussian(sigma_points_meas)
    Pxz = ukf._cross_covariance(
        sigma_points, sigma_points_meas, X_pred, z_pred)
    PzR = ukf.make_positive_definite(Pz+R)  
    inv_Pz_u = torch.cholesky_inverse(torch.linalg.cholesky(PzR))
    # inv_Pz_u = torch.cholesky_inverse(torch.linalg.cholesky(Pz + R))

    K = torch.matmul(Pxz, inv_Pz_u)

    X_updated = X_pred + \
        torch.matmul(K, (Z - z_pred).unsqueeze(-1)).squeeze(-1)
    P_updated = P_pred - \
        torch.matmul(K, torch.matmul(Pz, K.transpose(-2, -1)))

    return X_updated, P_updated

results = []
infer_times = []
count = 0

vehicle_model.eval()
sensor_model.eval()

with torch.no_grad():
    for x, x_next in test_dataloader:
        x = x.to(device)  # (B, seq_len, 10)
        x_next = x_next.to(device)  # (B,10)
        
        if count == 0:
            x_hat = x[:,-1,:5] # (B, 5)
            P= torch.diag(torch.tensor([1e-3, 1e-3, 1e-3, 1e-3, 1e-5],device=device)).unsqueeze(0)
            Q= torch.diag(torch.tensor([1e-3, 1e-3, 1e-2, 1e-2, 1e-5], device=device)).unsqueeze(0)
            R=torch.diag(torch.diag(torch.tensor([1e-1, 1e-1, 1e-1, 1e-1], device=device))).unsqueeze(0)

        # if count % 100 == 0:
        #     x_hat= x[-1]
        # time0 = time.time()
        time2 = time.time()
        imu_seq = x[:, :, -3:].permute(0, 2, 1)  # (B, 3, seq_len)
        imu=sensor_model(imu_seq)  # (B, 3)
        # imu=x[:, -1, 6:9]  # (B, 3)
        wheel_speed = x[:, -1, 3].unsqueeze(-1) # (B, 1)
        y = torch.cat((imu, wheel_speed), dim=-1) # (B, 4)
        u= x[:, -1, 5:7] # (B, 2) 
        # time1 = time.time()
        x_hat, P = predict(x_hat, P, Q, u)
        x_hat, P = update(x_hat, P, y, R, u)
        time3 = time.time()
        
        x_hat_np = x_hat.cpu().numpy().squeeze()
        target_np = x_next.cpu().numpy().squeeze()
        
        results.append({
            'prediction': x_hat_np,
            'target': target_np,
        })


        infer_time = np.clip(time3 - time2, 0, 0.01)
        infer_times.append(infer_time)
        count += 1


metrics = ['v_x', 'v_y', 'r']

# Create a 3×1 grid of subplots
fig, axs = plt.subplots(3, 1, sharex=True)
axs = axs.flat  # flatten to a 1D iterator

times = np.arange(0, (len(results)) * 0.01, 0.01).tolist()

# Loop over each metric/index
for idx, (ax, metric) in enumerate(zip(axs, metrics)):

    # ax.plot(times, [res['rmse'] for res in results], label='RMSE')
    ax.plot(times, [res['prediction'][idx] for res in results], label='Prediction')
    ax.plot(times, [res['target'][idx] for res in results], label='Target')
    ax.set_ylabel(f'{metric}')
    ax.set_xlabel('time')
    ax.legend()
    ax.grid()
    



times = np.arange(0, len(infer_times) * 0.01, 0.01).tolist()

plt.figure()
plt.plot(times, infer_times, label='Neural Pacejka')
plt.title(f'infer.mean: {np.median(infer_times):.4f} s')
plt.legend()
plt.xlabel('Batch index')
plt.ylabel('Time (s)')


plt.tight_layout()
plt.show()