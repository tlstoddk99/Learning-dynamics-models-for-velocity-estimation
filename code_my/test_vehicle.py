import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import torch

from robot_models.single_track_pacejka import SingleTrackPacejkaModel
from robot_models.single_track_parameters import SingleTrackParameters
from tire_models.pacejka import PacejkaTireModel


from torch.utils.data import DataLoader
# from torch.nn.utils import remove_weight_norm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths and logging setup
df = pd.read_csv('/home/a/Learning-dynamics-models-for-velocity-estimation/code_my/opti_test/hoons_all_test.csv', index_col=0)
# df = pd.read_csv('/home/a/Learning-dynamics-models-for-velocity-estimation/code_my/opti_test/hoons_all_train_and_val.csv', index_col=0)
timestamp = time.strftime('%m-%d_%H-%M')

# fix seed
torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

vehicle_parameters = SingleTrackParameters()
tire_model = PacejkaTireModel(vehicle_parameters)
model = SingleTrackPacejkaModel(vehicle_parameters, tire_model)
model.to(device)
model = torch.jit.script(model)
# model.load_state_dict(model_state_dict)
model = model.to(device)
model.eval()

#make dataloader

count = 0
results = []
infer_times = []

model.eval()
# with torch.no_grad():
#     for inputs,targets in test_dataloader:
#         # inputs=inputs.flatten(start_dim=1)
#         time0 =time.time()
#         mu, var = model(inputs)
#         time1 = time.time()
        
#         mu = mu.cpu().numpy().flatten()
#         var = var.cpu().numpy().flatten()
#         targets = targets.cpu().numpy().flatten()
        
#         std= np.sqrt(var)

#         # 3 sigma rule: 99.73% of the data
#         # 2 sigma rule: 95.45% of the data
#         # 1 sigma rule: 68.27% of the data
#         uncertainty = 2*std

#         results.append({
#             'prediction': mu,
#             'target': targets,
#             'error': (mu - targets),
#             'uncertainty': uncertainty,
#         })
        
#         if count >2:
#             infer_times.append(time1 - time0)
#         count += 1

# #plot the results
# #[ax,ay,r,wheel_speed]

# # Define your metrics in order
# # metrics = ['ax', 'ay', 'r', 'wheel_speed']
# metrics = ['ax', 'ay', 'r']

# # Create a 2×2 grid of subplots
# fig, axs = plt.subplots(3, 1)
# axs = axs.flat  # flatten to a 1D iterator

# times = np.arange(0, (len(results)) * 0.01, 0.01).tolist()

# # Loop over each metric/index
# for idx, (ax, metric) in enumerate(zip(axs, metrics)):
#     preds = [res['prediction'][idx] for res in results]
#     targs = [res['target'][idx]     for res in results]
#     uncs  = [res['uncertainty'][idx] for res in results]

#     ax.set_title(metric)
#     ax.set_xlabel('time')
#     ax.set_ylabel(metric)

#     ax.plot(times, preds, label='prediction')
#     ax.plot(times, targs, alpha=0.8 ,label='target')
#     ax.fill_between(
#         times,
#         [p - u for p, u in zip(preds, uncs)],
#         [p + u for p, u in zip(preds, uncs)],
#         alpha=0.2,
#         label='uncertainty'
#     )
#     ax.legend()
#     ax.grid()
    
# fig, axs_2 =  plt.subplots(3, 1)
# axs_2 = axs_2.flat  # flatten to a 1D iterator

# for idx, (ax_2, metric) in enumerate(zip(axs_2, metrics)):
#     errs = [res['error'][idx] for res in results]
#     uncs = [res['uncertainty'][idx] for res in results]
    
#     ax_2.set_title(f'{metric} error')
#     ax_2.set_xlabel('time')
#     ax_2.set_ylabel('error')
#     ax_2.plot(times, errs, label='error')
#     # ax_2.plot(times, uncs, label='uncertainty')
#     ax_2.fill_between(
#         times,
#         [u for u in uncs],
#         [-u for u in uncs],
#         alpha=0.2,
#         label='uncertainty',
#         color='C1'
#     )
#     ax_2.legend()
#     ax_2.grid()

# times = np.arange(0, len(infer_times) * 0.01, 0.01).tolist()

# plt.figure()
# plt.plot(times, infer_times)
# plt.title(f'mean inference time: {np.mean(infer_times):.4f} s')
# plt.xlabel('Batch index')
# plt.ylabel('Time (s)')


# plt.tight_layout()
# plt.show()
    
    
    
    
    

    
        
        
        
        
        
    
