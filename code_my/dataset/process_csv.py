import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from tqdm import tqdm


def lowpass_filter(data, cutoff, fs=100, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def find_optimal_cutoff(gt_signal, noisy_signal, fs=100, cutoff_range=np.linspace(0.1, 20, 100)):
    best_cutoff = None
    lowest_mse = float('inf')
    filtered_best = None

    for cutoff in cutoff_range:
        filtered = lowpass_filter(noisy_signal, cutoff, fs=fs)
        mse = mean_squared_error(gt_signal, filtered)
        if mse < lowest_mse:
            lowest_mse = mse
            best_cutoff = cutoff
            filtered_best = filtered

    return best_cutoff, filtered_best

def normalize_imu(ax,ay,r):
    """
    Normalize IMU signals to a range of [-1, 1]
    """
    a_scale = 50
    w_scale = 7
    ax = np.clip(ax, -a_scale, a_scale) / a_scale
    ay = np.clip(ay, -a_scale, a_scale) / a_scale
    r = np.clip(r, -w_scale, w_scale) / w_scale
    return ax, ay, r

def denormalize_imu(ax,ay,r):
    """
    Denormalize IMU signals from a range of [-1, 1]
    """
    a_scale = 50
    w_scale = 7
    ax = ax * a_scale
    ay = ay * a_scale
    r = r * w_scale
    return ax, ay, r

def preprocess_df(
    df: pd.DataFrame,
    dt: float = 0.01,
) -> pd.DataFrame:
    """
    1) Fill NaNs
    2) Compute IMU ground truth from motion-capture velocities
    3) Compute error signals (error = noise + bias)
    4) Separate noise and bias by applying a low-pass filter
    5) Compute noise and bias signals
    
    """
    # Compute GT IMU
    dvx = df["v_x"].diff().fillna(0.0) / dt
    dvy = df["v_y"].diff().fillna(0.0) / dt
    yaw_rate = df["r"]
    df["ax_gt"] = dvx - yaw_rate * df["v_y"]
    df["ay_gt"] = dvy + yaw_rate * df["v_x"]
    df["r_gt"] = yaw_rate
    
    # Normalize IMU signals
    df["ax_imu_norm"], df["ay_imu_norm"], df["r_imu_norm"] = normalize_imu(
        df["ax_imu"].to_numpy(),
        df["ay_imu"].to_numpy(),
        df["r_imu"].to_numpy()
    )
    # Normalize GT signals
    df["ax_gt_norm"], df["ay_gt_norm"], df["r_gt_norm"] = normalize_imu(
        df["ax_gt"].to_numpy(),
        df["ay_gt"].to_numpy(),
        df["r_gt"].to_numpy()
    )
    

    df["ax_fc"] = np.nan
    df["ay_fc"] = np.nan
    df["r_fc"] = np.nan
    
    window_size = 600
    half_window = window_size // 2
    if len(df) < window_size:
        return pd.DataFrame()

    for idx in tqdm(range(half_window, len(df) - half_window), desc=f"Processing {df['run_id'].iloc[0]}"):
        window_df = df.iloc[idx - half_window:idx + half_window]

        cutoff_ax, _ = find_optimal_cutoff(
            window_df["ax_gt"].to_numpy(),
            window_df["ax_imu"].to_numpy()
        )
        cutoff_ay, _ = find_optimal_cutoff(
            window_df["ay_gt"].to_numpy(),
            window_df["ay_imu"].to_numpy()
        )
        cutoff_r, _ = find_optimal_cutoff(
            window_df["r_gt"].to_numpy(),
            window_df["r_imu"].to_numpy()
        )

        df.at[idx, "ax_fc"] = cutoff_ax
        df.at[idx, "ay_fc"] = cutoff_ay
        df.at[idx, "r_fc"] = cutoff_r

    # Drop rows where optimal cutoff frequencies couldn't be computed
    df = df.dropna(subset=["ax_fc", "ay_fc", "r_fc"]).reset_index(drop=True)
    return df


def preprocess_group(group_df):
    return preprocess_df(group_df)







df= pd.read_csv("/home/a/Learning-dynamics-models-for-velocity-estimation/code_my/dataset/hoons_all_train_and_val.csv")
# df= pd.read_csv("/home/a/Learning-dynamics-models-for-velocity-estimation/code_my/dataset/hoons_all_test.csv")


processed_df = df.groupby("run_id", group_keys=False).apply(preprocess_group)

processed_df.to_csv("/home/a/Learning-dynamics-models-for-velocity-estimation/code_my/dataset/hoons_all_train_and_val_processed.csv", index=False)


# def fourier_transform(signal: np.ndarray, dt: float = 0.01):
#     """
#     Perform Fourier Transform on the signal and return frequency and magnitude.
#     """
#     n = len(signal)
#     freq = np.fft.fftfreq(n, d=dt)
#     fft_values = np.fft.fft(signal)
#     magnitude = np.abs(fft_values)
#     return freq[:n // 2], magnitude[:n // 2]

# def compare_fft(df, dt=0.01):
#     """
#     Plot FFT of ax, ay, r for IMU and GT data for comparison.
#     """
#     fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

#     signals = [("ax_imu", "ax_gt"), ("ay_imu", "ay_gt"), ("r_imu", "r_gt")]
#     titles = ["X Acceleration", "Y Acceleration", "Rotation"]

#     for i, ((imu_col, gt_col), title) in enumerate(zip(signals, titles)):
#         freq_imu, mag_imu = fourier_transform(df[imu_col].to_numpy(), dt)
#         freq_gt, mag_gt = fourier_transform(df[gt_col].to_numpy(), dt)
#         #mean filter window size: 500
#         mag_imu = np.convolve(mag_imu, np.ones(500)/500, mode='valid')
#         mag_gt = np.convolve(mag_gt, np.ones(500)/500, mode='valid')

#         # Adjust frequency axis to match the filtered magnitude
#         freq_imu = freq_imu[:len(mag_imu)]
#         freq_gt = freq_gt[:len(mag_gt)]

#         axs[i].plot(freq_imu, mag_imu, label="IMU")
#         axs[i].plot(freq_gt, mag_gt, label="Ground Truth")
#         axs[i].set_title(f"Frequency Spectrum: {title}")
#         axs[i].set_ylabel("Magnitude")
#         axs[i].grid(True)
#         axs[i].legend()
    
    
#     # signals = [("ax_e", "ax_e_b"), ("ay_e", "ay_e_b"), ("r_e", "r_e_b")]
#     # signals = [("ax_e"),("ay_e"),("r_e")]
#     # titles = ["X Error", "Y Error", "Rotation Error"]
#     # # for i, ((error_col, bias_col), title) in enumerate(zip(signals, titles)):
#     # for i, (error_col,title) in enumerate(zip(signals, titles)):
#     #     freq_error, mag_error = fourier_transform(df[error_col].to_numpy(), dt)
#     #     # freq_bias, mag_bias = fourier_transform(df[bias_col].to_numpy(), dt)

#     #     axs[i].plot(freq_error, mag_error, label="Error Signal")
#     #     # axs[i].plot(freq_bias, mag_bias, label="Bias Signal")
#     #     axs[i].set_title(f"Frequency Spectrum: {title}")
#     #     axs[i].set_ylabel("Magnitude")
#     #     axs[i].grid(True)
#     #     axs[i].legend()

#     axs[-1].set_xlabel("Frequency (Hz)")
#     plt.tight_layout()
#     plt.show()




# Example usage
# if __name__ == "__main__":
#     # Load your DataFrame here
#     df = pd.read_csv("/home/a/Learning-dynamics-models-for-velocity-estimation/code_my/dataset/hoons_all_train_and_val.csv")
#     # #normalize IMU signals
#     # df["ax_imu"], df["ay_imu"], df["r_imu"] = normalize_imu(
#     #     df["ax_imu"].to_numpy(),
#     #     df["ay_imu"].to_numpy(),
#     #     df["r_imu"].to_numpy()
#     # )
    
#     # Preprocess the DataFrame
#     df = preprocess_df(df)
    

    # fs = 100  # 샘플링 주파수

    # # 필터 함수 정의
    # def lowpass_filter(data, cutoff, fs, order=4):
    #     nyq = 0.5 * fs
    #     normal_cutoff = cutoff / nyq
    #     b, a = butter(order, normal_cutoff, btype='low', analog=False)
    #     return filtfilt(b, a, data)

    # # 평가할 신호 목록
    # signal_groups = [
    #     ("ax_imu", "ax_gt"),
    #     ("ay_imu", "ay_gt"),
    #     ("r_imu", "r_gt"),
    # ]

    # # 윈도우 후보 리스트
    # window_seconds_list = [5,5.5, 6, 6.5, 7]
    # cutoff_range = np.linspace(0.1, fs/2 - 1, 50)

    # # 전체 결과 저장
    # results = []

    # for win_sec in tqdm(window_seconds_list):
    #     window_size = int(win_sec * fs)
    #     step_size = fs  # 1초 간격

    #     metrics_all = []

    #     for signal_col, gt_col in signal_groups:
    #         signal = df[signal_col].to_numpy()
    #         gt = df[gt_col].to_numpy()

    #         filtered_variable = np.zeros_like(signal)
    #         weights = np.zeros_like(signal)

    #         for start in range(0, len(signal) - window_size + 1, step_size):
    #             end = start + window_size
    #             window_signal = signal[start:end]
    #             window_gt = gt[start:end]

    #             # 최적 cutoff 탐색
    #             min_mse = float("inf")
    #             best_cutoff = None
    #             for cutoff in cutoff_range:
    #                 try:
    #                     filtered_win = lowpass_filter(window_signal, cutoff, fs)
    #                     mse = mean_squared_error(window_gt, filtered_win)
    #                     if mse < min_mse:
    #                         min_mse = mse
    #                         best_cutoff = cutoff
    #                 except:
    #                     continue

    #             filtered_win = lowpass_filter(window_signal, best_cutoff, fs)
    #             filtered_variable[start:end] += filtered_win
    #             weights[start:end] += 1

    #         filtered_variable /= np.maximum(weights, 1)

    #         # 성능 측정
    #         mse = mean_squared_error(gt, filtered_variable)
    #         rmse = np.sqrt(mse)
    #         mae = mean_absolute_error(gt, filtered_variable)
    #         corr, _ = pearsonr(gt, filtered_variable)

    #         metrics_all.append((rmse, mae, corr))

    #     # 평균 성능 계산
    #     rmse_mean = np.mean([m[0] for m in metrics_all])
    #     mae_mean = np.mean([m[1] for m in metrics_all])
    #     corr_mean = np.mean([m[2] for m in metrics_all])

    #     results.append({
    #         "window_sec": win_sec,
    #         "rmse": rmse_mean,
    #         "mae": mae_mean,
    #         "corr": corr_mean
    #     })

    # # 결과 저장 및 출력
    # df_result = pd.DataFrame(results)
    # print(df_result)

    # # 최적 윈도우 선택
    # best_row = df_result.loc[df_result["rmse"].idxmin()]
    # print(f"\n✅ 최적의 윈도우 길이: {best_row['window_sec']}초 (평균 RMSE={best_row['rmse']:.5f}, Corr={best_row['corr']:.5f})")

    # # 시각화
    # x = df_result["window_sec"].values
    # plt.figure()
    # plt.plot(x, df_result["rmse"].values, label="Avg RMSE")
    # plt.xlabel("Window size (sec)")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    
    # df= df[8000:9000]
    # Compare FFT of IMU and GT signals
    # compare_fft(df)
    
    # import numpy as np
    # import pandas as pd
    # import matplotlib.pyplot as plt
    # from scipy.signal import butter, filtfilt
    # from sklearn.metrics import mean_squared_error, mean_absolute_error
    # from scipy.stats import pearsonr

    # # 신호 준비
    # signal = df["ax_imu"].to_numpy()
    # gt = df["ax_gt"].to_numpy()
    # fs = 100  # 샘플링 주파수

    # # 필터 함수 정의
    # def lowpass_filter(data, cutoff, fs, order=4):
    #     nyq = 0.5 * fs
    #     normal_cutoff = cutoff / nyq
    #     b, a = butter(order, normal_cutoff, btype='low', analog=False)
    #     return filtfilt(b, a, data)

    # # 윈도우 리스트 (초 단위)
    # window_seconds_list = np.arange(7.0, 7.2, 0.01)  
    # cutoff_range = np.linspace(0.1, fs/2 - 1, 50)

    # results = []

    # for win_sec in window_seconds_list:
    #     window_size = int(win_sec * fs)
    #     step_size = fs  # 1초 간격

    #     filtered_variable = np.zeros_like(signal)
    #     weights = np.zeros_like(signal)

    #     for start in range(0, len(signal) - window_size + 1, step_size):
    #         end = start + window_size
    #         window_signal = signal[start:end]
    #         window_gt = gt[start:end]

    #         # 최적 cutoff 탐색
    #         min_mse = float("inf")
    #         best_cutoff = None
    #         for cutoff in cutoff_range:
    #             try:
    #                 filtered_win = lowpass_filter(window_signal, cutoff, fs)
    #                 mse = mean_squared_error(window_gt, filtered_win)
    #                 if mse < min_mse:
    #                     min_mse = mse
    #                     best_cutoff = cutoff
    #             except:
    #                 continue

    #         # 필터 적용
    #         filtered_win = lowpass_filter(window_signal, best_cutoff, fs)
    #         filtered_variable[start:end] += filtered_win
    #         weights[start:end] += 1

    #     filtered_variable /= np.maximum(weights, 1)

    #     # 성능 평가
    #     mse = mean_squared_error(gt, filtered_variable)
    #     rmse = np.sqrt(mse)
    #     mae = mean_absolute_error(gt, filtered_variable)
    #     corr, _ = pearsonr(gt, filtered_variable)

    #     results.append({
    #         "window_sec": win_sec,
    #         "mse": mse,
    #         "rmse": rmse,
    #         "mae": mae,
    #         "corr": corr
    #     })

    # # 결과 테이블
    # df_result = pd.DataFrame(results)
    # print(df_result)

    # # 성능 지표 시각화
    # plt.figure(figsize=(12, 6))
    # x = df_result["window_sec"].values
    # plt.plot(x, df_result["rmse"].values, marker='o', label="RMSE")
    # plt.plot(x, df_result["mae"].values, marker='s', label="MAE")
    # plt.plot(x, df_result["corr"].values, marker='^', label="Corr")
    # plt.xlabel("Window size (sec)")
    # # plt.title("Adaptive LPF 성능 vs 윈도우 크기")
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    
    # best_row = df_result.loc[df_result["rmse"].idxmin()]
    # print(f"✅ 최적의 윈도우 길이: {best_row['window_sec']}초 (RMSE={best_row['rmse']:.5f}, Corr={best_row['corr']:.5f})")
 

    # #calc snr
    # snr_ax = 10 * np.log10(
    #     np.var(df["ax_gt"].to_numpy()) / np.var(df["ax_e"].to_numpy())
    # )
    # snr_ay = 10 * np.log10(
    #     np.var(df["ay_gt"].to_numpy()) / np.var(df["ay_e"].to_numpy())
    # )
    # snr_r = 10 * np.log10(
    #     np.var(df["r_gt"].to_numpy()) / np.var(df["r_e"].to_numpy())
    # )
    # print(f"SNR Ax: {snr_ax:.2f} dB")
    # print(f"SNR Ay: {snr_ay:.2f} dB")
    # print(f"SNR R: {snr_r:.2f} dB")
    
    # from scipy.signal import savgol_filter
    # bias_ax = savgol_filter(df["ax_e"], window_length=101, polyorder=3)
    # noise_ax = df["ax_e"] - bias_ax

    # snr_ax_denoised = 10 * np.log10(np.var(df["ax_gt"]) / np.var(noise_ax))
    # print(f"SNR Ax (Denoised): {snr_ax_denoised:.2f} dB")
    
    # import matplotlib.pyplot as plt

    # plt.plot(df["ax_e"], label="Error (IMU - GT)")
    # plt.plot(bias_ax, label="Estimated Bias (Savgol)")
    # # plt.plot(noise_ax, label="Residual Noise")
    # plt.legend()
    # plt.title("Bias and Noise Separation")
    # plt.show()
    
    


    # Create the dataset
    # dataset = IMUDataset(df,run_ids=[23])
    # dataset = IMUDataset()

    # for run_id in dataset.runs.keys():
    #     dataset.plot(run_id)
    
    # print(f"Dataset size: {len(dataset)}")
    # print(f"Input shape: {dataset.inputs.shape}")
    # print(f"Target shape: {dataset.targets.shape}")
    
    
    # print(f"ax_imu min/max: {dataset.inputs[:, 0, :].min()}, {dataset.inputs[:, 0, :].max()}")
    # print(f"ay_imu min/max: {dataset.inputs[:, 1, :].min()}, {dataset.inputs[:, 1, :].max()}")
    # print(f"r_imu min/max: {dataset.inputs[:, 2, :].min()}, {dataset.inputs[:, 2, :].max()}")
    # print(f"ax_bias min/max: {dataset.targets[:, 0].min()}, {dataset.targets[:, 0].max()}")
    # print(f"ay_bias min/max: {dataset.targets[:, 1].min()}, {dataset.targets[:, 1].max()}")
    # print(f"r_bias min/max: {dataset.targets[:, 2].min()}, {dataset.targets[:, 2].max()}")