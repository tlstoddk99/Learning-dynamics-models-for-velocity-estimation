import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import joblib 


class UkfDataset(torch.utils.data.Dataset):
    def __init__(self, df, seq_len=5, run_id_list=None):
        self.seq_len = seq_len
        self.features =  ['v_x', 'v_y', 'r', 'omega_wheels', 'friction', 'delta', 'Iq', 'ax_imu', 'ay_imu', 'r_imu']
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

            for t in range(seq_len - 1, len(df_run)-1):
                seq = df_run.loc[t - seq_len + 1: t, self.features].values # t- seq_len + 1 to t
                target = df_run.loc[t+1, self.features].values  # t + 1
                self.inputs.append(seq.astype(np.float32))
                self.targets.append(target.astype(np.float32))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = torch.tensor(self.inputs[idx])  # shape: (seq_len, n_features)
        x_next = torch.tensor(self.targets[idx])  # shape: (n_features,)
        return x, x_next
    
    
if __name__ == "__main__":
    df = pd.read_csv("/home/a/Learning-dynamics-models-for-velocity-estimation/code_my/dataset/hoons_all_test_gt.csv", index_col=0)
    # df = pd.read_csv('/home/a/Learning-dynamics-models-for-velocity-estimation/code_my/dataset/hoons_all_train_and_val_gt.csv', index_col=0)
    dataset = UkfDataset(df, seq_len=6, run_id_list=[4])
    print(f"Dataset length: {len(dataset)}")