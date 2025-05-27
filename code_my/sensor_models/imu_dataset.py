import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import joblib 
class IMUDataset(torch.utils.data.Dataset):
    def __init__(self, df, seq_len=5, run_id_list=None):
        self.seq_len = seq_len
        self.imu_features = ['ax_imu', 'ay_imu', 'r_imu']
        self.gt_features = ['ax_gt', 'ay_gt', 'r_gt']

        self.inputs = []
        self.targets = []

        if run_id_list is None:
            run_ids = df['run_id'].unique()
        else:
            run_ids = run_id_list

        for run_id in run_ids:
            df_run = df[df['run_id'] == run_id].reset_index(drop=True)
            if len(df_run) < seq_len:
                continue

            for t in range(seq_len - 1, len(df_run)):
                seq = df_run.loc[t - seq_len + 1: t, self.imu_features].values.T  # shape: (channels, seq_len)
                target = df_run.loc[t, self.gt_features].values  # shape: (n_targets,)
                self.inputs.append(seq.astype(np.float32))
                self.targets.append(target.astype(np.float32))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = torch.tensor(self.inputs[idx])  # shape: (channels, seq_len)
        y = torch.tensor(self.targets[idx])  # shape: (n_targets,)
        return x, y

if __name__ == "__main__":
    df = pd.read_csv("/home/a/Learning-dynamics-models-for-velocity-estimation/code_my/dataset/hoons_all_train_and_val_gt.csv")
    imu_features = ['ax_imu', 'ay_imu', 'r_imu']
    gt_features = ['ax_gt', 'ay_gt', 'r_gt']

    # 평균과 표준편차 계산
    imu_mean = df[imu_features].mean()
    imu_std = df[imu_features].std()
    gt_mean = df[gt_features].mean()
    gt_std = df[gt_features].std()

    # 정규화 적용
    df[imu_features] = (df[imu_features] - imu_mean) / imu_std
    df[gt_features] = (df[gt_features] - gt_mean) / gt_std

    # 정규화 파라미터 저장
    norm_params = {
        'imu_mean': imu_mean,
        'imu_std': imu_std,
        'gt_mean': gt_mean,
        'gt_std': gt_std
    }
    joblib.dump(norm_params, 'normalization_params.pkl')

    # dataset = IMUDataset(df, seq_len=3)
