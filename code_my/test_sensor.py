import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import torch
import pickle
from sklearn.preprocessing import StandardScaler
from sensor_models.debias_model import TCNGaussian
from sensor_models.imu_model import LiteTCNGaussian
from sensor_models.de_bias_dataset import DeBiasDataset
from sensor_models.imu_dataset import IMUDataset, preprocess_df, denormalize_imu
from torch.utils.data import DataLoader


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths and logging setup
df = pd.read_csv('/home/a/Learning-dynamics-models-for-velocity-estimation/code_my/dataset/hoons_all_test.csv', index_col=0)
# df = pd.read_csv('/home/a/Learning-dynamics-models-for-velocity-estimation/code_my/dataset/hoons_all_train_and_val.csv', index_col=0)

# Load the model state dict
model_state_dict = torch.load(
    '/home/a/Learning-dynamics-models-for-velocity-estimation/code_my/trained_models/05-23_17-46/best_epoch_994_loss_-2.9642.pt',
                              )
timestamp = time.strftime('%m-%d_%H-%M')

# fix seed
torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)



df = preprocess_df(df)
test_dateset = IMUDataset(df,device=device)
test_dataloader = DataLoader(test_dateset, batch_size=1, shuffle=False)


# model = TCNGaussian()
model = LiteTCNGaussian()
model.to(device)
model = torch.jit.script(model)
model.load_state_dict(model_state_dict)
model = model.to(device)

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
        
       # GPU → CPU → numpy, 유지하고 싶은 차원만 flatten 하지 않기
        mu_np     = mu .cpu().numpy()    # shape (B,3)
        var_np    = var.cpu().numpy()    # shape (B,3)
        targ_np   = targets.cpu().numpy()# shape (B,3)

        # ----- Denormalize -----
        # 1) model 예측
        ax_mu, ay_mu, r_mu = denormalize_imu(mu_np[:,0],
                                             mu_np[:,1],
                                             mu_np[:,2])
        mu_denorm = np.stack([ax_mu, ay_mu, r_mu], axis=1).flatten()  # (B,3)

        # 2) 실제 타겟
        ax_t, ay_t, r_t = denormalize_imu(targ_np[:,0],
                                          targ_np[:,1],
                                          targ_np[:,2])
        targ_denorm = np.stack([ax_t, ay_t, r_t], axis=1).flatten()  # (B,3)

        # 3) 불확실성: var → std → denormalize
        std_np = np.sqrt(var_np)   # (B,3)
        ax_s, ay_s, r_s = denormalize_imu(std_np[:,0],
                                          std_np[:,1],
                                          std_np[:,2])
        std_denorm = np.stack([ax_s, ay_s, r_s], axis=1).flatten()  # (B,3)
       

        # 3 sigma rule: 99.73% of the data
        # 2 sigma rule: 95.45% of the data
        # 1 sigma rule: 68.27% of the data
        uncertainty = 2 * std_denorm   # (B,3)
        

        results.append({
            'prediction': mu_denorm,
            'target': targ_denorm,
            'error': (mu_denorm - targ_denorm),
            'uncertainty': uncertainty,
        })
        
        if count >2:
            infer_times.append(time1 - time0)
        count += 1

#plot the results
#[ax,ay,r,wheel_speed]
# Define your metrics in order
# metrics = ['ax', 'ay', 'r', 'wheel_speed']
metrics = ['ax_bias', 'ay_bias', 'r_bias']

# Create a 2×2 grid of subplots
fig, axs = plt.subplots(3, 1)
axs = axs.flat  # flatten to a 1D iterator

times = np.arange(0, (len(results)) * 0.01, 0.01).tolist()

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
    
fig, axs_2 =  plt.subplots(3, 1)
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
    
    
    
    
    

    
        
        
        
        
        
    
