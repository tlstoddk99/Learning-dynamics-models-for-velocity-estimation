import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import torch
import pickle
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import torchdiffeq as ode

# Local imports
from robot_models.vehicle_dataset import VehicleDataset
from robot_models.single_track_pacejka import SingleTrackPacejkaModel
from robot_models.pacejka import PacejkaTireModel
from robot_models.single_track_parameters import SingleTrackParameters


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths and logging setup
df = pd.read_csv('/home/a/Learning-dynamics-models-for-velocity-estimation/code_my/dataset/hoons_all_test_gt.csv', index_col=0)
# df = pd.read_csv('/home/a/Learning-dynamics-models-for-velocity-estimation/code_my/dataset/hoons_all_train_and_val.csv', index_col=0)

# Load the model state dict
model_state_dict = torch.load(
    '/home/a/Learning-dynamics-models-for-velocity-estimation/code_my/trained_models/vehicle/05-27_01-26/best_epoch_184_loss_2.2829.pt',
                              )
timestamp = time.strftime('%m-%d_%H-%M')



# fix seed
torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

test_dataset = VehicleDataset(df)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

vehicle_params = SingleTrackParameters()
tire_model = PacejkaTireModel(vehicle_parameters=vehicle_params)

model= SingleTrackPacejkaModel(
    vehicle_parameters=vehicle_params,
    tire_model=tire_model
)


# Load the model state dict
# model.load_state_dict(model_state_dict)

model = torch.jit.script(model)
state_weights = torch.tensor([0.2225, 0.5064, 0.1566, 0.1145, 0.0, 0.0, 0.0], device=device)
model.to(device)

results = []
infer_times = []
count = 0
model.eval()
with torch.no_grad():
    for x, x_next in test_dataloader:
        rmses= []
        
        x = x.to(device)  # (B,7)
        x_next = x_next.to(device)  # (B,7)
        
        time2 = time.time()
        pred_x = ode.odeint(
            model,
            x,
            torch.tensor([0, 0.01], device=x.device),
            method="rk4",
            rtol=1e-5,
            atol=1e-6
        )[-1]
        time3 = time.time()
        
        # pred_x = pred_x* state_weights  # Apply state weights

        x_np = x.cpu().numpy().squeeze()
        pred_x_np = pred_x.cpu().numpy().squeeze()
        target_np = x_next.cpu().numpy().squeeze()
        # Calculate RMSE for each state variable
        rmse = np.sqrt(np.mean((pred_x_np - target_np) ** 2, axis=0))
            
        
        
        results.append({
            'input': x_np,
            'prediction': pred_x_np,
            'target': target_np,
            'rmse': rmse,
        })


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
    
    
    
    
    

    
        
        
        
        
        
    
