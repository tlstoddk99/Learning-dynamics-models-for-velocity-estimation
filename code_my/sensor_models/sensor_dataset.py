import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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

        # print(self.batches.shape)

        last_run = int(self.batches.shape[0] * dataset_scaler)
        self.batches = self.batches[0:last_run, :, :]

        self.dt = torch.tensor([0.0, self.Ts], dtype=dtype, device=device)

    @staticmethod
    def state_def():
        return ['v_x', 'v_y', 'r', 'omega_wheels', 
                'friction', 'delta', 'Iq', 
                'ax_imu', 'ay_imu', 'r_imu',
                'ax', 'ay','ax_next', 'ay_next', 'r_next', 'omega_wheels_next']

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
            # if the number of samples is not divisible by self.sequence_length, drop the last samples
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
        
    def get_imu_bias(self,data):
        """
        Get the IMU bias from the data
        bias=sum(raw-gt)/N
        """

class IMUBiasFrictionDataset(torch.utils.data.Dataset):
    """
    A dataset that uses the past 5 seconds (500 samples) of IMU and wheel speed data as input,
    and predicts the average IMU biases (ax, ay, yaw-rate) and average friction coefficient
    over the next 1 second (100 samples).

    Input DataFrame columns:
        - v_x, v_y, r          : body velocities and yaw rate (motion capture)
        - omega_wheels         : wheel speed
        - delta, Iq            : steering angle and motor current
        - ax_imu, ay_imu, r_imu : raw IMU measurements
        - friction             : friction coefficient during the test segment
        - run_id               : experiment/run ID

    Arguments:
        input_seq_len: length of the input sequence
        pred_seq_len : length of the prediction window
        dt           : time interval between samples
    """
    def __init__(
        self,
        df: pd.DataFrame,
        input_seq_len: int = 500,
        pred_seq_len: int = 100,
        dt: float = 0.01,
        dtype=torch.float32,
        device=torch.device("cpu")
    ):
        self.input_seq_len = input_seq_len
        self.pred_seq_len = pred_seq_len
        self.dt = dt
        self.dtype = dtype
        self.device = device
        
        self.debiased_imu = None

        df = self._fill_na(df)
        runs = self._split_runs(df)
        for run_id, run_df in runs.items():
            self.plot_imu(run_df)
        
        # for run_id, run_df in runs.items():
        #     print(f"Processing run_id: {run_id}")
        #     print(f"Run length: {len(run_df)}\n")
        
        inputs, targets = self._generate_samples(runs)

        self.inputs = torch.tensor(
            np.stack(inputs), dtype=self.dtype, device=self.device
        )  # [N, input_seq_len, 4]
        self.targets = torch.tensor(
            np.stack(targets), dtype=self.dtype, device=self.device
        )  # [N, 4]
        

    def _fill_na(self, df: pd.DataFrame) -> pd.DataFrame:
        # Fill any missing values with zero
        if df.isnull().any().any():
            print("Data contains NaN values. Filling NaN values with 0.")
            return df.fillna(0)
        return df

    def _split_runs(self, df: pd.DataFrame) -> dict:
        """
        Split the DataFrame by run_id and reset each group's index.
        """
        return {
            rid: grp.reset_index(drop=True)
            for rid, grp in df.groupby("run_id")
        }

    def _generate_samples(self, runs: dict) -> tuple:
        """
        Generate input and target samples for all runs.
        """
        all_inputs, all_targets = [], []
        for run_df in runs.values():
            ins, tars = self._process_single_run(run_df)
            all_inputs.extend(ins)
            all_targets.extend(tars)
        return all_inputs, all_targets

    def _process_single_run(self, run_df: pd.DataFrame) -> tuple:
        """
        Generate samples within a single run using a sliding window approach.
        """
        inputs, targets = [], []
        total_len = len(run_df)
        step = self.pred_seq_len
        for start in range(
            0,
            total_len - self.input_seq_len - self.pred_seq_len + 1,
            step
        ):
            inp = self._get_input_window(run_df, start)
            tar = self._get_target_window(run_df, start)
            inputs.append(inp)
            targets.append(tar)

        return inputs, targets

    def _get_input_window(self, df: pd.DataFrame, start: int) -> np.ndarray:
        # Extract input window: [ax_imu, ay_imu, r_imu, omega_wheels]
        cols = ["ax_imu", "ay_imu", "r_imu", "omega_wheels"]
        return df.loc[start : start + self.input_seq_len - 1, cols].values

    def _get_target_window(
        self, df: pd.DataFrame, start: int
    ) -> list:
        """
        Compute the mean IMU biases and friction coefficient over the prediction window.
        """
        pred = df.loc[
            start + self.input_seq_len :
            start + self.input_seq_len + self.pred_seq_len - 1
        ]

        # Calculate average IMU biases
        mean_ax, mean_ay, mean_r = self._compute_mean_biases(pred)

        # Calculate average friction coefficient
        friction_mean = pred["friction"].mean()

        return [mean_ax, mean_ay, mean_r, friction_mean]

    def _compute_mean_biases(self, pred: pd.DataFrame) -> tuple:
        """
        Compute the average difference between true accelerations/angular rate
        and IMU measurements over the prediction window.
        """
        # Estimate true accelerations from motion capture data
        dvx = pred["v_x"].diff().fillna(0.0) / self.dt
        dvy = pred["v_y"].diff().fillna(0.0) / self.dt
        psi = pred["r"]

        ax_true = dvx - psi * pred["v_y"]
        ay_true = dvy + psi * pred["v_x"]
        r_true = pred["r"]

        # Calculate biases as differences between IMU readings and true values
        ax_bias = pred["ax_imu"] - ax_true
        ay_bias = pred["ay_imu"] - ay_true
        r_bias = pred["r_imu"] - r_true
        return ax_bias.mean(), ay_bias.mean(), r_bias.mean()

    def plot_imu(self, runs):
        """
        plot the IMU sensor data and the ground truth data
        """
        # 전체 길이, 시간축
        n = len(runs)
        time = np.arange(n) * self.dt

        # 1) pandas Series → NumPy array 변환
        ax_imu = runs["ax_imu"].to_numpy()
        ay_imu = runs["ay_imu"].to_numpy()
        r_imu  = runs["r_imu"].to_numpy()

        # 2) ground truth 계산 (Series 연산 후 to_numpy 으로 변환)
        dvx = runs["v_x"].diff().fillna(0.0).to_numpy() / self.dt
        dvy = runs["v_y"].diff().fillna(0.0).to_numpy() / self.dt
        psi = runs["r"].to_numpy()
        ax_true = (dvx - psi * runs["v_y"].to_numpy())
        ay_true = (dvy + psi * runs["v_x"].to_numpy())
        r_true  = runs["r"].to_numpy()

        # 3) bias 배열 미리 만들기
        pred = self.pred_seq_len
        ax_bias = np.zeros(n)
        ay_bias = np.zeros(n)
        r_bias  = np.zeros(n)

        # 4) 섹션별 평균 bias 계산 후 NumPy 배열에 채우기
        for start in range(0, n, pred):
            end = min(start + pred, n)
            ax_b = runs["ax_imu"].iloc[start:end].mean() - ax_true[start:end].mean()
            ay_b = runs["ay_imu"].iloc[start:end].mean() - ay_true[start:end].mean()
            r_b  = runs["r_imu"].iloc[start:end].mean()  - r_true[start:end].mean()
            ax_bias[start:end] = ax_b
            ay_bias[start:end] = ay_b
            r_bias[start:end]   = r_b

        # 5) debiased signal
        debiased_ax = ax_imu - ax_bias
        debiased_ay = ay_imu - ay_bias
        debiased_r  = r_imu  - r_bias

        # 6) 플롯
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 6))
        fig.suptitle(f"Run ID: {runs['run_id'].iloc[0]}")
        axs[0].set_ylabel("ax")
        axs[0].plot(time, ax_imu,      label="ax_imu")
        axs[0].plot(time, ax_true,     label="ax_true")
        axs[0].plot(time, debiased_ax, label="debiased_ax")
        axs[0].step(time, ax_bias, where='post', linestyle='--', label="ax_bias")
        axs[0].legend()

        axs[1].set_ylabel("ay")
        axs[1].plot(time, ay_imu,      label="ay_imu")
        axs[1].plot(time, ay_true,     label="ay_true")
        axs[1].plot(time, debiased_ay, label="debiased_ay")
        axs[1].step(time, ay_bias, where='post', linestyle='--', label="ay_bias")
        axs[1].legend()

        axs[2].set_ylabel("r")
        axs[2].plot(time, r_imu,       label="r_imu")
        axs[2].plot(time, r_true,      label="r_true")
        axs[2].plot(time, debiased_r,  label="debiased_r")
        axs[2].step(time, r_bias, where='post', linestyle='--', label="r_bias")
        axs[2].set_xlabel("Time (s)")
        axs[2].legend()

        plt.tight_layout()
        plt.show()
        

    def __len__(self):
        # Return the number of samples
        return self.inputs.size(0)

    def __getitem__(self, idx: int):
        # Retrieve a single sample (input sequence and corresponding target)
        return self.inputs[idx], self.targets[idx]



if __name__ == "__main__":
    df= pd.read_csv("/home/a/Learning-dynamics-models-for-velocity-estimation/code/opti_test/hoons_all_train_and_val.csv")
    # df= pd.read_csv("/home/a/Learning-dynamics-models-for-velocity-estimation/code/opti_test/hoons_all_test.csv")
    dataset=IMUBiasFrictionDataset(df)
