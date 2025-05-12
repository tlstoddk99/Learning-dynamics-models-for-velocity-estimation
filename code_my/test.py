import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import torch
from sensor_models.sensor_refine_model import TCNGaussian
from sensor_models.sensor_dataset import SensorDataset
from utils.argparser import get_parser




def parse_args():
    parser = get_parser()
    return parser.parse_args()

args = parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths and logging setup
df = pd.read_csv('/home/a/Learning-dynamics-models-for-velocity-estimation/code_my/opti_test/hoons_all_test.csv', index_col=0)
timestamp = time.strftime('%m-%d_%H-%M')


# Create the dataset
test_dataset = SensorDataset(
    df,
    subsample_all=1,
    Ts_multiplier=1,
    check_new_run=True,
    test_run_id=[],
    test=False,
    dtype=torch.float32 if args.common_precision == 32 else torch.float64,
    device=device,
    dataset_scaler=1.0,
    # sequence_length=args.ukf_sequence_length,
    sequence_length=100,
)
test_data_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1, shuffle=False
)

print(f"test dataloader shape: {test_dataset.batches.shape}")


#create the model
model = TCNGaussian(
    input_size=4,
    output_size=4,
    num_channels=64,
    num_levels=4,
    kernel_size=2,
    dropout=0.2,
    activation=torch.nn.ReLU,
    eps=1e-3
)
model.to(device)



# Load the model state dict
model_state_dict = torch.load('/home/a/Learning-dynamics-models-for-velocity-estimation/code_my/trained_models/05-12_16-04/best_epoch_2797_loss_2.0435.pt')

model.load_state_dict(model_state_dict)
model.eval()



# errors = []
# predictions = []
# targets = []
# uncertainties = []

results = []
infer_times = []

model.eval()
with torch.no_grad():
    for batch in test_data_loader:
        imu = batch[:,7:10,:]
        wheel = batch[:,3:4,:]
        inputs = torch.cat((imu, wheel), dim=1).to(device)
        time0 =time.time()
        mu, var = model(inputs)
        time1 = time.time()
        
        targets = batch[:, -4:, -1]
        
        mu = mu.cpu().numpy().flatten()
        var = var.cpu().numpy().flatten()
        targets = targets.cpu().numpy().flatten()
        
        uncertainty = np.sqrt(var)
        
       
        
        results.append({
            'prediction': mu,
            'target': targets,
            'error': mu - targets,
            'uncertainty': uncertainty,
        })
        
        infer_times.append(time1 - time0)
        

#plot the results
#[ax,ay,r,wheel_speed]

# Define your metrics in order
metrics = ['ax', 'ay', 'r', 'wheel_speed']

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
    ax.plot(times, targs, label='target')
    ax.fill_between(
        times,
        [p - u for p, u in zip(preds, uncs)],
        [p + u for p, u in zip(preds, uncs)],
        alpha=0.2,
        label='uncertainty'
    )
    ax.legend()

plt.tight_layout()


plt.figure()
plt.plot(times, infer_times)
plt.title(f'mean inference time: {np.mean(infer_times):.4f} s')
plt.xlabel('Batch index')
plt.ylabel('Time (s)')



plt.show()
    
    
    
    
    

    
        
        
        
        
        
    
