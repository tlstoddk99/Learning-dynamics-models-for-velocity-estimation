import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import torch
import pickle
from sklearn.preprocessing import StandardScaler
from sensor_models.imu_model import ImuModel
from sensor_models.imu_dataset import IMUDataset
from torch.utils.data import DataLoader


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths and logging setup
df = pd.read_csv('/home/a/Learning-dynamics-models-for-velocity-estimation/code_my/dataset/hoons_all_test_gt.csv', index_col=0)
# df = pd.read_csv('/home/a/Learning-dynamics-models-for-velocity-estimation/code_my/dataset/hoons_all_train_and_val.csv', index_col=0)

# Load the model state dict
model_state_dict = torch.load(
    '/home/a/Learning-dynamics-models-for-velocity-estimation/code_my/trained_models/05-26_22-48/best_epoch_199_loss_1.5399.pt',
                              )
timestamp = time.strftime('%m-%d_%H-%M')

# fix seed
torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

test_dataset = IMUDataset(df)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = ImuModel()
model.to(device)
model = torch.jit.script(model)
model.load_state_dict(model_state_dict)
model = model.to(device)

model.eval()

results = []
infer_times = []
count = 0
model.eval()
with torch.no_grad():
    for inputs,targets in test_dataloader:
        
        inputs = inputs.to(device)  # (B,3,seq_len)
        targets = targets.to(device)  # (B,3)
        # inputs=inputs.flatten(start_dim=1)
        time0 =time.time()
        x = model(inputs)
        time1 = time.time()

        inputs = inputs[-1, :, -1].cpu().numpy().squeeze()  # (3, seq_len) -> (3,)
        x_np = x.cpu().numpy().squeeze()  # (,3)
        target_np = targets.cpu().numpy().squeeze()  # (,3)

        results.append({
            'input': inputs,
            'prediction': x_np,
            'target': target_np,
            
        })

        infer_time = np.clip(time1 - time0, 0, 0.01)  # clip to 10ms
        infer_times.append(infer_time)


metrics = ['ax', 'ay', 'r']

# Create a 2×2 grid of subplots
fig, axs = plt.subplots(3, 1, sharex=True)
axs = axs.flat  # flatten to a 1D iterator

times = np.arange(0, (len(results)) * 0.01, 0.01).tolist()

# Loop over each metric/index
for idx, (ax, metric) in enumerate(zip(axs, metrics)):
    raws= [res['input'][idx] for res in results]
    preds = [res['prediction'][idx] for res in results]
    targs = [res['target'][idx]     for res in results]
    # uncs  = [res['uncertainty'][idx] for res in results]

    ax.set_title(metric)
    ax.set_xlabel('time')
    ax.set_ylabel(metric)

    ax.plot(times, preds, label='prediction')
    ax.plot(times, targs, alpha=0.8 ,label='target')
    ax.plot(times, raws, alpha=0.5, label='raw input')

    ax.legend()
    ax.grid()
    

times = np.arange(0, len(infer_times) * 0.01, 0.01).tolist()

plt.figure()
plt.plot(times, infer_times)
plt.title(f'mean inference time: {np.median(infer_times):.4f} s')
plt.xlabel('Batch index')
plt.ylabel('Time (s)')


plt.tight_layout()
plt.show()
    
    
    
    
    

    
        
        
        
        
        
    
