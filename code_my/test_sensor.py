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
from torch.utils.data import Subset
from sklearn.metrics import mean_squared_error


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths and logging setup
df = pd.read_csv('/home/a/Learning-dynamics-models-for-velocity-estimation/code_my/dataset/hoons_all_test_gt.csv', index_col=0)
# df = pd.read_csv('/home/a/Learning-dynamics-models-for-velocity-estimation/code_my/dataset/hoons_all_train_and_val.csv', index_col=0)


# Load the model state dict
model_state_dict = torch.load(
    '/home/a/Learning-dynamics-models-for-velocity-estimation/code_my/trained_models/05-29_22-00/best_epoch_3790_loss_1.2455.pt',
                              )
timestamp = time.strftime('%m-%d_%H-%M')

# fix seed
torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
test_dataset = IMUDataset(df,run_id_list=[4])
test_dataset = Subset(test_dataset, np.arange(3500, 4500, 1))  

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


# 각 센서 채널별 결과 저장용
from sklearn.metrics import mean_squared_error
import numpy as np
metrics = [r'$a_x$ [m/s$^2$]', r'$a_y$ [m/s$^2$]', r'$r$ [rad/s$^2$]']

rmse_raws = []
rmse_preds = []
snr_raws = []
snr_preds = []

for i in range(3):  # 각 채널
    raw = np.array([res['input'][i] for res in results])
    pred = np.array([res['prediction'][i] for res in results])
    targ = np.array([res['target'][i] for res in results])
    
    # RMSE
    rmse_raw = np.sqrt(mean_squared_error(targ, raw))
    rmse_pred = np.sqrt(mean_squared_error(targ, pred))

    # SNR
    signal_power = np.mean(targ ** 2)
    noise_power_raw = np.mean((targ - raw) ** 2)
    noise_power_pred = np.mean((targ - pred) ** 2)

    snr_raw = 10 * np.log10(signal_power / noise_power_raw) if noise_power_raw > 0 else np.inf
    snr_pred = 10 * np.log10(signal_power / noise_power_pred) if noise_power_pred > 0 else np.inf

    rmse_raws.append(rmse_raw)
    rmse_preds.append(rmse_pred)
    snr_raws.append(snr_raw)
    snr_preds.append(snr_pred)

# 평균 계산
avg_rmse_raw = np.mean(rmse_raws)
avg_rmse_pred = np.mean(rmse_preds)
avg_snr_raw = np.mean(snr_raws)
avg_snr_pred = np.mean(snr_preds)

# 출력
print("📊 Performance Comparison (Raw vs Model):\n")
for i, label in enumerate(metrics):
    print(f"{label}")
    print(f"  Raw       -> RMSE: {rmse_raws[i]:.4f}, SNR: {snr_raws[i]:.2f} dB")
    print(f"  Predicted -> RMSE: {rmse_preds[i]:.4f}, SNR: {snr_preds[i]:.2f} dB")
    print()

print("📈 Overall Averages:")
print(f"  Raw       -> Avg RMSE: {avg_rmse_raw:.4f}, Avg SNR: {avg_snr_raw:.2f} dB")
print(f"  Predicted -> Avg RMSE: {avg_rmse_pred:.4f}, Avg SNR: {avg_snr_pred:.2f} dB")


# metrics = [r'$a_x$ [m/s$^2$]', r'$a_y$ [m/s$^2$]', r'$r$ [rad/s]']
# fig, axs = plt.subplots(3, 1, figsize=(6, 6), sharex=True, dpi=300)
# axs = axs.flat

# times = np.arange(0, len(results) * 0.01, 0.01)

# for idx, (ax, metric_label) in enumerate(zip(axs, metrics)):
#     raws = [res['input'][idx] for res in results]
#     preds = [res['prediction'][idx] for res in results]
#     targs = [res['target'][idx] for res in results]

#     ax.plot(times, raws, label='Raw Sensor', color='C0', linewidth=0.5, alpha=0.5)
#     ax.plot(times, targs, label='Ground Truth', color='black', linewidth=0.6, alpha=0.8)
#     ax.plot(times, preds, label='Proposed', color='red', linewidth=0.7)

#     ax.set_ylabel(metric_label, fontsize=10)
#     ax.grid(True, linewidth=0.5, alpha=0.7)
#     # x축 눈금 설정: 1초 간격
#     ax.set_xticks(np.arange(0, times[-1] + 1, 1.0))

# # 마지막 subplot만 x축 라벨 추가
# axs[-1].set_xlabel('Time [s]', fontsize=10)

# # 공통 범례를 하단에 하나만 표시
# handles, labels = axs[0].get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=11, bbox_to_anchor=(0.55, 1.06))
# # fig.subplots_adjust(hspace=0.1, bottom=0.01)

# plt.tight_layout()
# plt.savefig(f'/home/a/Learning-dynamics-models-for-velocity-estimation/code_my/plots/{timestamp}_sensor_output.png', dpi=600, bbox_inches='tight')

# times = np.arange(0, len(infer_times) * 0.01, 0.01).tolist()

# plt.figure()
# plt.plot(times, infer_times)
# plt.title(f'mean inference time: {np.median(infer_times):.4f} s')
# plt.xlabel('Batch index')
# plt.ylabel('Time (s)')


# plt.tight_layout()
# plt.show()
    
    
    
    
    

    
        
        
        
        
        
    
