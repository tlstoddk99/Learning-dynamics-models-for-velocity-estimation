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
from filters.ukf_model_steper_inference import UKFModelStepperInference
from robot_models.single_track_pacejka import SingleTrackPacejkaModel
from robot_models.pacejka import PacejkaTireModel
from robot_models.single_track_parameters import SingleTrackParameters
from sensor_models.imu_model import ImuModel


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths and logging setup
df = pd.read_csv('/home/a/Learning-dynamics-models-for-velocity-estimation/code_my/opti_test/hoons_all_test_gt.csv', index_col=0)


# Load the model state dict
sensor_model_path = torch.load(
    '/home/a/Learning-dynamics-models-for-velocity-estimation/code_my/trained_models/05-26_22-48/best_epoch_199_loss_1.5399.pt',
                              )
dataset= UkfDataset(df)
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

sensor_model = ImuModel()
sensor_model.load_state_dict(sensor_model_path)

model = UKFModelStepperInference(vehicle_model=vehicle_model,sensor_model=sensor_model, dt=0.01)
sensor_model.to(device)
model.to(device)
sensor_model = torch.jit.script(sensor_model)
model = torch.jit.script(model)


results = []
infer_times = []
count = 0
model.eval()
with torch.no_grad():
    for x, x_next in test_dataloader:
        x = x.to(device)  # (B, seq_len, 10)
        x_next = x_next.to(device)  # (B,10)
        if count == 0:
            x_hat = x[-1]
            P=model.diag_P0
        imu_seq = x[:, :, -3:].permute(0, 2, 1)  # (B, 3, seq_len)
        imu=sensor_model(imu_seq)  # (B, 3)
        wheel_speed = x[:, -1, 3].unsqueeze(-1) # (B, 1)
        y = torch.cat((imu, wheel_speed), dim=-1) # (B, 4)
        u= x[:, -1, 5:7]  # (B, 1, 2)
        # if count % 100 == 0:
        #     x_hat= x[-1]
        
        
        time2 = time.time()
        x_hat, P = model(x_hat=x_hat, P=P, u=u, y=y)
        time3 = time.time()
        
        # pred_x = pred_x* state_weights  # Apply state weights

        # x_np = x.cpu().numpy().squeeze()
        # pred_x_np = pred_x.cpu().numpy().squeeze()
        # target_np = x_next.cpu().numpy().squeeze()
        # # Calculate RMSE for each state variable
        # rmse = np.sqrt(np.mean((pred_x_np - target_np) ** 2, axis=0))
            
        
        
        # results.append({
        #     'input': x_np,
        #     'prediction': pred_x_np,
        #     'target': target_np,
        #     'rmse': rmse,
        # })


        infer_time = np.clip(time3 - time2, 0, 0.01)
        infer_times.append(infer_time)


metrics = ['v_x', 'v_y', 'r', 'omega_wheels', 'friction']

# Create a 5×1 grid of subplots
fig, axs = plt.subplots(5, 1, sharex=True)
axs = axs.flat  # flatten to a 1D iterator

times = np.arange(0, (len(results)) * 0.01, 0.01).tolist()

# Loop over each metric/index
for idx, (ax, metric) in enumerate(zip(axs, metrics)):

    ax.plot(times, [res['rmse'] for res in results], label='RMSE')
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