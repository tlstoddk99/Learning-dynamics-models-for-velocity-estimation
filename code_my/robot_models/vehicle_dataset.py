import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

class VehicleDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, run_ids= None):
        self.df = df
        # self.state_features = ['v_x', 'v_y', 'r', 'omega_wheels', 'friction', 'delta', 'Iq', 'ax_imu', 'ay_imu', 'r_imu']
        self.state_features = ['v_x', 'v_y', 'r', 'omega_wheels', 'friction', 'delta', 'Iq']
        self.x = []
        self.x_next = []
        
        if run_ids is None:
            run_ids = df['run_id'].unique()

        for run_id in run_ids:
            df_run = df[df['run_id'] == run_id].reset_index(drop=True)
            for t in range(0, len(df_run) - 2):
                x = df_run.loc[t, self.state_features].values
                x_next = df_run.loc[t+1, self.state_features].values
                self.x.append(x.astype(np.float32))
                self.x_next.append(x_next.astype(np.float32))
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx])  # shape: (channels,)
        x_next = torch.tensor(self.x_next[idx])  # shape: (channels,)
        return x, x_next


if __name__ == "__main__":
    df = pd.read_csv("/home/a/Learning-dynamics-models-for-velocity-estimation/code_my/dataset/hoons_all_train_and_val_gt.csv")
    
