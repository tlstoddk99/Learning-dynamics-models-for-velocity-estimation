import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import torch
from code_my.sensor_models.debias_model import TCNGaussian
# from sensor_models.sensor_dataset import SensorDataset
from sensor_models.de_bias_dataset import DeBiasDataset
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths and logging setup
df = pd.read_csv('/home/a/Learning-dynamics-models-for-velocity-estimation/code_my/opti_test/hoons_all_test.csv', index_col=0)
timestamp = time.strftime('%m-%d_%H-%M')

# fix seed
torch.manual_seed(42)
np.random.seed(42)

test_dateset = DeBiasDataset(
    df,
    device=device
)
test_dataloader = DataLoader(
    test_dateset,
    batch_size=1,
    shuffle=False,
)

model = TCNGaussian(
        input_size=3,
        output_size=3,
        dropout=0.2,
        activation=torch.nn.SiLU
    )
model.to(device)

# Load the model state dict
# model_state_dict = torch.load('/home/a/Learning-dynamics-models-for-velocity-estimation/code_my/trained_models/05-12_19-37/best_epoch_2384_loss_1.6559.pt')
model_state_dict = torch.load(
    '/home/a/Learning-dynamics-models-for-velocity-estimation/code_my/trained_models/05-19_17-26/best_epoch_2000_loss_1.2820.pt'
                              )
model.eval()

count = 0
results = []
infer_times = []

model.eval()
with torch.no_grad():
    for inputs,targets in test_dataloader:
        time0 =time.time()
        mu, var = model(inputs)
        time1 = time.time()
        
        mu = mu.cpu().numpy().flatten()
        var = var.cpu().numpy().flatten()
        targets = targets.cpu().numpy().flatten()
        
        # 3 sigma rule: 99.73% of the data
        # 2 sigma rule: 95.45% of the data
        # 1 sigma rule: 68.27% of the data
        uncertainty = 2*np.sqrt(var)
        
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
    
    
    
    
    

    
        
        
        
        
        
    
