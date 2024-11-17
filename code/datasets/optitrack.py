import torch
import numpy as np
import pandas as pd


class OptitrackDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        csv_file,
        subsample_all=1,
        Ts_multiplier=1,
        check_new_run=True,
        test_run_id=None,
        test=False,
        dtype=torch.float32,
        dataset_scaler=1.0,
        device=torch.device("cpu"),
    ):
        self.Ts_multiplier = Ts_multiplier
        self.check_new_run = check_new_run
        self.all_data = pd.read_csv(csv_file, index_col=0)

        self.all_data = self.all_data.iloc[::subsample_all, :]

        test_index = self.all_data["run_id"].isin(test_run_id)
        train_index = ~test_index
        self.all_data = self.all_data[test_index if test else train_index]

        self.data = self.all_data[self.state_def()]

        self.run_id = self.all_data["run_id"].values

        self.t = self.all_data.index.values

        self.data_torch = torch.tensor(
            self.data.values, dtype=dtype, device=device)

        last_idx = int(self.data_torch.shape[0] * dataset_scaler)
        self.data_torch = self.data_torch[:last_idx, :]

        dts = np.diff(self.t)

        # assert np.abs(dts.mean() - dts.max()) < 1e-3, \
        #     'check if the data is sampled at constant rate which is required by the ODE solver'

        self.Ts = np.median(dts) * self.Ts_multiplier
        print(f'test: {test} Ts = {self.Ts}')

        self.dt = torch.tensor([0.0, self.Ts], dtype=dtype, device=device)

    @staticmethod
    def state_def():
        return ['v_x', 'v_y', 'r', 'omega_wheels', 'friction', 'delta', 'Iq', 'ax_imu', 'ay_imu', 'r_imu']

    def __len__(self):
        return self.data_torch.shape[0] - self.Ts_multiplier

    def __getitem__(self, idx):
        while (
            self.check_new_run
            and self.run_id[idx] != self.run_id[idx + self.Ts_multiplier]
        ):
            idx += 1
            idx %= self.__len__()
        x = self.data_torch[idx]
        x_next = self.data_torch[idx + self.Ts_multiplier]
        return x, x_next

    def plot(self):
        self.data.plot(
            subplots=True, figsize=(10, 10), grid=True, title="Raw data", sharex=True
        )

    def get_state_weights_diff(self):
        w = ((1 / (torch.diff(self.data_torch, dim=0).abs()).mean(dim=0))[:4])
        w = w / w.sum()
        return w

    def get_min_values(self):
        return self.data_torch.min(dim=0).values

    def get_max_values(self):
        return self.data_torch.max(dim=0).values
