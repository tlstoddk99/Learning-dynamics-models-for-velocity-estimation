import torch
import pandas as pd
import numpy as np

class SensorDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df,
        subsample_all=1,
        Ts_multiplier=1,
        check_new_run=True,
        test_run_id=None,
        test=False,
        dtype=torch.float32,
        device=torch.device("cpu"),
        dataset_scaler=1.0,
        sequence_length=256,
    ):
        """
        Create a dataset for the sensor model
        dataframe should contain the following columns:
        input: [ax_imu,ay_imu,r_imu,wheel_speed]
        - (batch, channel, seq_len)
        output: [ax_gt,ay_gt,r_gt,wheel_speed_gt]
        - (batch, output_size)
        """
        self.Ts_multiplier = Ts_multiplier
        self.check_new_run = check_new_run
        self.sequence_length = sequence_length
        self.device = device
        self.dtype = dtype
        
        self.all_data = df
        
        self.all_data = self.all_data.iloc[::subsample_all, :]

        test_index = self.all_data["run_id"].isin(test_run_id)
        train_index = ~test_index
        self.all_data = self.all_data[test_index if test else train_index]

        self.t = self.all_data.index.values
        dts = np.diff(self.t)

        # assert np.abs(dts.mean() - dts.max()) < 1e-3, \
        #     'check if the data is sampled at constant rate which is required by the ODE solver'

        self.Ts = np.median(dts) * self.Ts_multiplier
        print(f'Ts = {self.Ts}')
        
        self.generate_gt()
        self.data = self.all_data[self.state_def()]

        self.run_id = self.all_data["run_id"].values

        self.generate_sequences()

        print(self.batches.shape)

        last_run = int(self.batches.shape[0] * dataset_scaler)
        self.batches = self.batches[0:last_run, :, :]

        self.dt = torch.tensor([0.0, self.Ts], dtype=dtype, device=device)

    @staticmethod
    def state_def():
        return ['v_x', 'v_y', 'r', 'omega_wheels', 'friction', 'delta', 'Iq', 'ax_imu', 'ay_imu', 'r_imu','ax', 'ay','ax_next', 'ay_next', 'r_next', 'omega_wheels_next']

    def __len__(self):
        return self.batches.shape[0]

    def __getitem__(self, idx):
        # """[B,L,C]"""
        # return self.batches[idx]
        # """[B,C,L]"""
        return self.batches[idx].permute(1,0)
        
    def plot(self):
        self.data.plot(
            subplots=True, figsize=(10, 10), grid=True, title="Raw data", sharex=True
        )

    def generate_sequences(self):
        """
        for every run id:
            - find all indexes of this run
            - split them into batches of size self.batch_size (if there are not enough samples in the run, skip it)
            - add batches to self.batches [sequences, len, n_states]
        """
        batches = []

        for run_id in np.unique(self.run_id):
            run_indexes = np.where(self.run_id == run_id)[0]
            if len(run_indexes) < self.sequence_length:
                continue

            run_indexes = run_indexes[
                : len(run_indexes) // self.sequence_length * self.sequence_length
            ]
            batches_idxs = np.array_split(
                run_indexes, len(run_indexes) // self.sequence_length
            )

            for batch_idxs in batches_idxs:
                batch = torch.tensor(
                    self.data.values[batch_idxs, :],
                    dtype=self.dtype,
                    device=self.device,
                )

                batches.append(batch)

        self.batches = torch.stack(batches)
        


    def generate_gt(self):
        """
        Generate the ground truth next step data:
          ax = Δv_x / Δt(next time step)
          ay = Δv_y / Δt(next time step)
          wheel_speed = wheel_speed(next time step)
        """
        df = self.all_data

        # compute Δv per run and divide by constant Ts
        df['ax'] = df.groupby('run_id')['v_x'].diff() / self.Ts
        df['ay'] = df.groupby('run_id')['v_y'].diff() / self.Ts
        
        
        df['ax_next'] = df.groupby('run_id')['ax'].shift(-1)
        df['ay_next'] = df.groupby('run_id')['ay'].shift(-1)
        df['r_next'] = df.groupby('run_id')['r'].shift(-1)
        df['omega_wheels_next'] = df.groupby('run_id')['omega_wheels'].shift(-1)

        # for the first sample of each run, fill NaN → 0.0
        df.fillna({'ax': 0.0, 'ay': 0.0,'ax_next':0.0,'ay_next':0.0,'r_next':0.0,'omega_wheels_next':0.0}, inplace=True)

        # write back
        self.all_data = df
        

