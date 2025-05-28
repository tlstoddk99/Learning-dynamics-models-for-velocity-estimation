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
from torch.utils.data import Subset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths and logging setup
df = pd.read_csv('/home/a/Learning-dynamics-models-for-velocity-estimation/code_my/dataset/hoons_all_test_gt.csv', index_col=0)

start_idx = 1000
# Load the model state dict
sensor_model_path = torch.load(
    '/home/a/Learning-dynamics-models-for-velocity-estimation/code_my/trained_models/05-26_22-48/best_epoch_199_loss_1.5399.pt',
                              )
dataset= UkfDataset(df,run_id_list=[4])
dataset = Subset(dataset, np.arange(start_idx, start_idx+2000, 1))  
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
    # PzR = ukf.make_positive_definite(Pz+R)  
    # inv_Pz_u = torch.cholesky_inverse(torch.linalg.cholesky(PzR))
    inv_Pz_u = torch.cholesky_inverse(torch.linalg.cholesky(Pz + R))

    K = torch.matmul(Pxz, inv_Pz_u)

    X_updated = X_pred + \
        torch.matmul(K, (Z - z_pred).unsqueeze(-1)).squeeze(-1)
    P_updated = P_pred - \
        torch.matmul(K, torch.matmul(Pz, K.transpose(-2, -1)))

    return X_updated, P_updated

results = []
raw_infer_times = []
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
            x_hat_raw = x[:,-1,:5] # (B, 5)
            P= torch.diag(torch.tensor([1e-3, 1e-3, 1e-3, 1e-3, 1e-5],device=device)).unsqueeze(0)
            P_raw= torch.diag(torch.tensor([1e-3, 1e-3, 1e-3, 1e-3, 1e-3], device=device)).unsqueeze(0)
            Q= torch.diag(torch.tensor([1e-3, 1e-3, 1e-2, 1e-2, 1e-5], device=device)).unsqueeze(0)
            R=torch.diag(torch.tensor([50e-1, 10e-1, 10e-1, 1e-1], device=device)).unsqueeze(0)
            R_raw=torch.diag(torch.tensor([5e1, 5e1, 5e1, 1e-1], device=device)).unsqueeze(0)


        wheel_speed = x[:, -1, 3].unsqueeze(-1) # (B, 1)
        u= x[:, -1, 5:7] # (B, 2) 
        
        imu_raw=x[:, -1, 6:9]  # (B, 3)
        time0 = time.time()
        y_raw = torch.cat((imu_raw, wheel_speed), dim=-1) # (B, 4)
        x_hat_raw, P_raw = predict(x_hat_raw, P_raw, Q, u)
        x_hat_raw, P_raw = update(x_hat_raw, P_raw, y_raw, R_raw, u)
        time1 = time.time()
        
        
        imu_seq = x[:, :, -3:].permute(0, 2, 1)  # (B, 3, seq_len)
        time2 = time.time()
        imu=sensor_model(imu_seq)  # (B, 3)
        y = torch.cat((imu, wheel_speed), dim=-1) # (B, 4)
        x_hat, P = predict(x_hat, P, Q, u)
        x_hat, P = update(x_hat, P, y, R, u)
        time3 = time.time()
        
        x_hat_np = x_hat.cpu().numpy().squeeze()
        x_hat_raw_np = x_hat_raw.cpu().numpy().squeeze()
        target_np = x_next.cpu().numpy().squeeze()
        
        results.append({
            'Raw Sensor': x_hat_raw_np,
            'Proposed': x_hat_np,
            'Error Raw Sensor': x_hat_raw_np - target_np[:5],
            'Error Proposed': x_hat_np - target_np[:5],
            'GT': target_np,
        })


        raw_infer_time = np.clip(time1 - time0, 0, 0.02)
        infer_time = np.clip(time3 - time2, 0, 0.02)  
        raw_infer_times.append(raw_infer_time)
        infer_times.append(infer_time)
        count += 1


# # 초기값
# x_init, y_init, yaw_init = 0.0, 0.0, 1.5
# dt = 0.01

# # 시간 생성
# times = np.arange(0, len(results) * dt, dt).tolist()

# def compute_trajectory(results, key):
#     x, y, yaw = x_init, y_init, yaw_init
#     positions = [(x, y)]
#     yaws = [yaw]

#     for res in results:
#         v_x = res[key][0]  # 차량 기준 전방 속도
#         v_y = res[key][1]  # 차량 기준 측면 속도
#         r = res[key][2]    # 요각 속도 (rad/s)

#         dx = v_x * np.cos(yaw) - v_y * np.sin(yaw)
#         dy = v_x * np.sin(yaw) + v_y * np.cos(yaw)

#         x += dx * dt
#         y += dy * dt
#         yaw += r * dt

#         positions.append((x, y))
#         yaws.append(yaw)

#     return np.array(positions), yaws

# # 세 궤적 계산
# positions_proposed, yaws_proposed = compute_trajectory(results, 'Proposed')
# positions_raw, yaws_raw = compute_trajectory(results, 'Raw Sensor')
# positions_gt, yaws_gt = compute_trajectory(results, 'GT')

# # 궤적 플로팅
# fig, ax = plt.subplots()
# ax.plot(positions_gt[:, 0], positions_gt[:, 1], label='Ground Truth', linewidth=4, alpha=0.7, color='black')
# ax.plot(positions_raw[:, 0], positions_raw[:, 1], label='Raw Sensor', linewidth=4, alpha=0.7, color='C0')
# ax.plot(positions_proposed[:, 0], positions_proposed[:, 1], label='Proposed', linewidth=4, alpha=0.7, color='red')

# arrow_interval = int(0.2 / dt)

# # 화살표 그리는 함수
# def draw_arrows(ax, positions, yaws, color):
#     for i in range(0, len(positions), arrow_interval):
#         px, py = positions[i]
#         yaw_i = yaws[i]
#         arrow_dx = np.cos(yaw_i) * 0.1
#         arrow_dy = np.sin(yaw_i) * 0.1
#         ax.arrow(px, py, arrow_dx, arrow_dy, head_width=0.1, head_length=0.1, width=0.04, color=color)

# # 각 궤적에 화살표 추가
# draw_arrows(ax, positions_gt, yaws_gt, 'black')
# draw_arrows(ax, positions_raw, yaws_raw, 'C0')
# draw_arrows(ax, positions_proposed, yaws_proposed, 'red')

# # 그래프 꾸미기
# ax.set_xlabel('X position [m]')
# ax.set_ylabel('Y position [m]')
# ax.grid(alpha=0.7)
# ax.set_aspect('equal', adjustable='box')
# ax.axis('equal')
# ax.legend()
# save_path = f'/home/a/Learning-dynamics-models-for-velocity-estimation/code_my/plots/ukf_pose/{start_idx}.png'
# plt.savefig(save_path, dpi=300, bbox_inches='tight')
# plt.show()



Q_n = Q.cpu().numpy().squeeze()
R_n = R.cpu().numpy().squeeze()
R_raw_n = R_raw.cpu().numpy().squeeze()

metrics = ['v_x', 'v_y', 'r']
errors=[]
# Create a 3×1 grid of subplots
fig, axs = plt.subplots(3, 1, sharex=True)
axs = axs.flat  # flatten to a 1D iterator

times = np.arange(0, (len(results)) * 0.01, 0.01).tolist()

# Loop over each metric/index
for idx, (ax, metric) in enumerate(zip(axs, metrics)):
    gt= [res['GT'][idx] for res in results]
    raw_sensor= [res['Raw Sensor'][idx] for res in results]
    proposed= [res['Proposed'][idx] for res in results]
    error_raw_sensor = [res['Error Raw Sensor'][idx] for res in results]
    error_proposed = [res['Error Proposed'][idx] for res in results]
    
    rmse= np.sqrt(np.mean(np.square(np.array(gt) - np.array(proposed))))
    errors.append(rmse)
    ax.plot(times, gt, label='GT', linewidth=2, alpha=0.7, color='black')
    ax.plot(times, raw_sensor, label='Raw Sensor', linewidth=2, alpha=0.7, color='C0')
    ax.plot(times, proposed, label='Proposed', linewidth=2, alpha=0.7, color='red')
    # ax.plot(times, error_raw_sensor, label='Error Raw Sensor', linewidth=2, alpha=0.7, color='C0')
    # ax.plot(times, error_proposed, label='Error Proposed', linewidth=2, alpha=0.7, color='red')
    ax.set_ylabel(f'{metric}')
    ax.set_xlabel('time')
    # ax.set_title(f'{metric} - Q: {Q_n[idx, idx]:.1e}, R: {R_n[idx, idx]:.1e}, R_raw: {R_raw_n[idx, idx]:.1e}')
    ax.set_title(f'{metric} - R: {R_n[idx, idx]:.3f}, E: {rmse:.3f}')
    ax.legend()
    ax.grid()
    
plt.tight_layout()
plt.savefig(f'/home/a/Learning-dynamics-models-for-velocity-estimation/code_my/plots/ukf_vel/{np.mean(errors):.3f}.png', dpi=300, bbox_inches='tight')
plt.show()


# plt.figure()
# plt.plot(times, infer_times, label='Proposed Inference Time', color='red', linewidth=2)
# plt.plot(times, raw_infer_times, label='Raw Sensor Inference Time', color='C0', linewidth=2)
# plt.xlabel('Batch Index')
# plt.ylabel('Inference Time (s)')
# plt.title(f'Mean Inference Time: {np.mean(infer_times):.4f} s (Proposed), {np.mean(raw_infer_times):.4f} s (Raw Sensor)')
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.show()