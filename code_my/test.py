import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import torch
import pickle
from sklearn.preprocessing import StandardScaler
# from sensor_models.sensor_dataset import SensorDataset
from sensor_models.debias_model import TCNGaussian
from sensor_models.debias_mlp_model import MLPGaussian
from sensor_models.de_bias_dataset import DeBiasDataset
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths and logging setup
df = pd.read_csv('/home/a/Learning-dynamics-models-for-velocity-estimation/code_my/opti_test/hoons_all_test.csv', index_col=0)
# df = pd.read_csv('/home/a/Learning-dynamics-models-for-velocity-estimation/code_my/opti_test/hoons_all_train_and_val.csv', index_col=0)

# Load the model state dict
model_state_dict = torch.load(
    '/home/a/Learning-dynamics-models-for-velocity-estimation/code_my/trained_models/05-20_01-41/best_epoch_2000_loss_0.9525.pt',
                              )
input_scaler_path = '/home/a/Learning-dynamics-models-for-velocity-estimation/code_my/trained_models/05-20_01-41/input_scaler.pkl'
target_scaler_path = '/home/a/Learning-dynamics-models-for-velocity-estimation/code_my/trained_models/05-20_01-41/target_scaler.pkl'
timestamp = time.strftime('%m-%d_%H-%M')

# fix seed
torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)


with open(input_scaler_path, 'rb') as f:
    input_scaler = pickle.load(f)
with open(target_scaler_path, 'rb') as f:
    target_scaler = pickle.load(f)

test_dateset = DeBiasDataset(
        df,
        device=device
    )

def flatten(ds):
    X = ds.inputs.permute(0, 2, 1).reshape(-1, ds.inputs.size(2)).cpu().numpy()
    y = ds.targets.cpu().numpy()
    return X, y

def scale_dataset(ds):
    X2d, y2d = flatten(ds)
    Xs = input_scaler.transform(X2d)
    ys = target_scaler.transform(y2d)
    N, seq_len, feat = ds.inputs.size()
    # reshape back to (N, feat, seq_len), then permute to (N, seq_len, feat)
    ds.inputs  = torch.tensor(Xs.reshape(N, feat, seq_len), dtype=torch.float32, device=device)\
                        .permute(0, 2, 1)
    ds.targets = torch.tensor(ys,dtype=torch.float32, device=device)

scale_dataset(test_dateset)

test_dataloader = DataLoader(test_dateset, batch_size=1, shuffle=False)

model = TCNGaussian(
        input_size=3,
        output_size=3,
        dropout=0.2,
        activation=torch.nn.SiLU
    )
# model= MLPGaussian(
#     activation=torch.nn.SiLU,
# )
model.to(device)




model.eval()

count = 0
results = []
infer_times = []

model.eval()
with torch.no_grad():
    for inputs,targets in test_dataloader:
        # inputs=inputs.flatten(start_dim=1)
        time0 =time.time()
        mu, var = model(inputs)
        time1 = time.time()
        
        mu = mu.cpu().numpy()
        var = var.cpu().numpy()
        targets = targets.cpu().numpy()
        
        mu_orig      = target_scaler.inverse_transform(mu.reshape(-1, 1)).flatten()
        targets_orig = target_scaler.inverse_transform(targets.reshape(-1, 1)).flatten()

     
        scale = target_scaler.scale_[0]  
        var_orig = var * (scale ** 2)

        std_orig = np.sqrt(var_orig)
       
        
        # 3 sigma rule: 99.73% of the data
        # 2 sigma rule: 95.45% of the data
        # 1 sigma rule: 68.27% of the data
        uncertainty = 2*std_orig
        
        results.append({
            'prediction': mu,
            'target': targets,
            'error': (mu - targets),
            'uncertainty': uncertainty,
        })
        if count >2:
            infer_times.append(time1 - time0)
        count += 1

#plot the results
#[ax,ay,r,wheel_speed]

# Define your metrics in order
# metrics = ['ax', 'ay', 'r', 'wheel_speed']
metrics = ['ax', 'ay', 'r']

# Create a 2×2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.flat  # flatten to a 1D iterator

# Prepare your data once
times = np.arange(0, len(results) * 0.01, 0.01).tolist()

# Loop over each metric/index
for idx, (ax, metric) in enumerate(zip(axs, metrics)):
    preds = [res['prediction'][idx] for res in results]
    targs = [res['target'][idx]     for res in results]
    uncs  = [res['uncertainty'][idx] for res in results]

    ax.set_title(metric)
    ax.set_xlabel('time')
    ax.set_ylabel(metric)

    ax.plot(times, preds, label='prediction')
    ax.plot(times, targs, alpha=0.8 ,label='target')
    ax.fill_between(
        times,
        [p - u for p, u in zip(preds, uncs)],
        [p + u for p, u in zip(preds, uncs)],
        alpha=0.2,
        label='uncertainty'
    )
    ax.legend()
    ax.grid()
    
fig, axs_2 = plt.subplots(2, 2, figsize=(12, 8))
axs_2 = axs_2.flat  # flatten to a 1D iterator

for idx, (ax_2, metric) in enumerate(zip(axs_2, metrics)):
    errs = [res['error'][idx] for res in results]
    uncs = [res['uncertainty'][idx] for res in results]
    
    ax_2.set_title(f'{metric} error')
    ax_2.set_xlabel('time')
    ax_2.set_ylabel('error')
    ax_2.plot(times, errs, label='error')
    # ax_2.plot(times, uncs, label='uncertainty')
    ax_2.fill_between(
        times,
        [u for u in uncs],
        [-u for u in uncs],
        alpha=0.2,
        label='uncertainty',
        color='C1'
    )
    ax_2.legend()
    ax_2.grid()

times = np.arange(0, len(infer_times) * 0.01, 0.01).tolist()

plt.figure()
plt.plot(times, infer_times)
plt.title(f'mean inference time: {np.mean(infer_times):.4f} s')
plt.xlabel('Batch index')
plt.ylabel('Time (s)')


plt.tight_layout()
plt.show()
    
    
    
    
    

    
        
        
        
        
        
    
