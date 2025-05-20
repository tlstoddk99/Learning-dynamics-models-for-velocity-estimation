import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

class DeBiasDataset(torch.utils.data.Dataset):
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
        step: int = 10,
        run_ids: list = None,
        dt: float = 0.01,
        dtype=torch.float32,
        device=torch.device("cpu"),
        plot=False,
    ):
        self.input_seq_len = input_seq_len
        self.pred_seq_len = pred_seq_len
        self.dt = dt
        self.dtype = dtype
        self.device = device
        self.run_ids = run_ids
        self.plot = plot
        self.step = step

        # Fill missing values
        df = self._fill_na(df)
        # Split DataFrame by run
        runs = self._split_runs(df)
        # Optionally plot IMU biases per run
        for run_id, run_df in runs.items():
            if plot and len(run_df) > self.input_seq_len + self.pred_seq_len:
                print(f"run {run_id}: {len(run_df)} samples")
                self.plot_imu(run_df)

        # Generate inputs and targets
        inputs, targets = self._generate_samples(runs)
        # Convert to torch tensors
        self.inputs = torch.tensor(
            np.stack(inputs), dtype=self.dtype, device=self.device
        )
        self.targets = torch.tensor(
            np.stack(targets), dtype=self.dtype, device=self.device
        )

    def _fill_na(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.isnull().any().any():
            print("Data contains NaN values. Filling NaN values with 0.")
            return df.fillna(0)
        return df

    def _split_runs(self, dataframe: pd.DataFrame) -> dict:
        split_dict = {}
        if self.run_ids is not None:
            for run_id in self.run_ids:
                run_df = dataframe[dataframe["run_id"] == run_id]
                split_dict[run_id] = run_df.reset_index(drop=True)
        else:
            for run_id in dataframe["run_id"].unique():
                run_df = dataframe[dataframe["run_id"] == run_id]
                split_dict[run_id] = run_df.reset_index(drop=True)
        return split_dict

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        a_scale = 50
        w_scale = 7
        df["ax_imu"] = np.clip(df["ax_imu"], -a_scale, a_scale)
        df["ay_imu"] = np.clip(df["ay_imu"], -a_scale, a_scale)
        df["r_imu"]  = np.clip(df["r_imu"], -w_scale, w_scale)
        df["ax_imu"] = (df["ax_imu"] + a_scale) / (2 * a_scale)
        df["ay_imu"] = (df["ay_imu"] + a_scale) / (2 * a_scale)
        df["r_imu"]  = (df["r_imu"] + w_scale) / (2 * w_scale)
        return df

    def _generate_samples(self, runs: dict) -> tuple:
        all_inputs, all_targets = [], []
        for run_df in runs.values():
            ins, tars = self._process_single_run(run_df)
            all_inputs.extend(ins)
            all_targets.extend(tars)
        return all_inputs, all_targets

    def _process_single_run(self, run_df: pd.DataFrame) -> tuple:
        inputs, targets = [], []
        total_len = len(run_df)
        for start in range(0, total_len - self.input_seq_len - self.pred_seq_len + 1, self.step):
            inp = self._get_input_window(run_df, start)
            tar = self._get_target_window(run_df, start)
            inputs.append(inp)
            targets.append(tar)
        return inputs, targets

    def _get_input_window(self, df: pd.DataFrame, start: int) -> np.ndarray:
        cols = ["ax_imu", "ay_imu", "r_imu"]
        input_df = self._preprocess_data(df.copy())
        data = input_df.loc[start : start + self.input_seq_len - 1, cols].values
        return data.T

    def _get_target_window(self, df: pd.DataFrame, start: int) -> list:
        pred = df.loc[
            start + self.input_seq_len :
            start + self.input_seq_len + self.pred_seq_len - 1
        ]
        mean_ax, mean_ay, mean_r = self._compute_mean_biases(pred)
        return [mean_ax, mean_ay, mean_r]

    def _compute_mean_biases(self, pred: pd.DataFrame) -> tuple:
        # Compute ground truth IMU measurements for the prediction window
        gt = self.make_gt_imu(pred)
        ax_true, ay_true, r_true = gt[:, 0], gt[:, 1], gt[:, 2]
        ax_bias = pred["ax_imu"].to_numpy() - ax_true
        ay_bias = pred["ay_imu"].to_numpy() - ay_true
        r_bias  = pred["r_imu"].to_numpy() - r_true
        return ax_bias.mean(), ay_bias.mean(), r_bias.mean()

    def plot_imu(self, run_df: pd.DataFrame):
        n = len(run_df)
        time = np.arange(n) * self.dt
        # Raw IMU readings
        ax_imu = run_df["ax_imu"].to_numpy()
        ay_imu = run_df["ay_imu"].to_numpy()
        r_imu  = run_df["r_imu"].to_numpy()
        # Ground truth signals via helper
        gt = self.make_gt_imu(run_df)
        ax_true, ay_true, r_true = gt[:, 0], gt[:, 1], gt[:, 2]
        # Prepare bias arrays
        pred_len = self.pred_seq_len
        ax_bias = np.zeros(n)
        ay_bias = np.zeros(n)
        r_bias  = np.zeros(n)
        # Segment-wise average
        for start in range(0, n, pred_len):
            end = min(start + pred_len, n)
            ax_b = run_df["ax_imu"].iloc[start:end].mean() - ax_true[start:end].mean()
            ay_b = run_df["ay_imu"].iloc[start:end].mean() - ay_true[start:end].mean()
            r_b  = run_df["r_imu"].iloc[start:end].mean()  - r_true[start:end].mean()
            ax_bias[start:end] = ax_b
            ay_bias[start:end] = ay_b
            r_bias[start:end]   = r_b
        # Debiased signals for visualization (optional)
        debiased_ax = ax_imu - ax_bias
        debiased_ay = ay_imu - ay_bias
        debiased_r  = r_imu  - r_bias
        # Plot
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 6))
        fig.suptitle(f"Run ID: {run_df['run_id'].iloc[0]}")
        axs[0].set_ylabel("ax bias")
        axs[0].step(time, ax_bias, where='post', label="ax_bias")
        axs[0].legend()
        axs[1].set_ylabel("ay bias")
        axs[1].step(time, ay_bias, where='post', label="ay_bias")
        axs[1].legend()
        axs[2].set_ylabel("yaw-rate bias")
        axs[2].step(time, r_bias, where='post', label="r_bias")
        axs[2].set_xlabel("Time (s)")
        axs[2].legend()
        plt.tight_layout()
        plt.show()

    def make_gt_imu(self, df: pd.DataFrame) -> np.ndarray:
        # Compute true accelerations and yaw-rate from motion capture
        dvx = df["v_x"].diff().fillna(0.0).to_numpy() / self.dt
        dvy = df["v_y"].diff().fillna(0.0).to_numpy() / self.dt
        psi = df["r"].to_numpy()
        ax_true = dvx - psi * df["v_y"].to_numpy()
        ay_true = dvy + psi * df["v_x"].to_numpy()
        r_true  = psi
        return np.column_stack((ax_true, ay_true, r_true))

    def __len__(self):
        return self.inputs.size(0)

    def __getitem__(self, idx: int):
        return self.inputs[idx], self.targets[idx]


if __name__ == "__main__":
    df = pd.read_csv("/home/a/Learning-dynamics-models-for-velocity-estimation/code/opti_test/hoons_all_train_and_val.csv")
    dataset = DeBiasDataset(df, plot=True)
    print(f"dataset shape: {df.shape}")
    print(f"inputs shape: {dataset.inputs.shape}")
    print(f"targets shape: {dataset.targets.shape}")
    print(f"inputs: {dataset.inputs[0, :, :]}")
    print(f"targets: {dataset.targets[0, :]}")
    print()
    print(f"minmax ax_imu: {dataset.inputs[:, 0, :].min()}, {dataset.inputs[:, 0, :].max()}")
    print(f"minmax ay_imu: {dataset.inputs[:, 1, :].min()}, {dataset.inputs[:, 1, :].max()}")
    print(f"minmax r_imu: {dataset.inputs[:, 2, :].min()}, {dataset.inputs[:, 2, :].max()}")
    print()
    print(f"minmax ax_bias: {dataset.targets[:, 0].min()}, {dataset.targets[:, 0].max()}")
    print(f"minmax ay_bias: {dataset.targets[:, 1].min()}, {dataset.targets[:, 1].max()}")
    print(f"minmax r_bias: {dataset.targets[:, 2].min()}, {dataset.targets[:, 2].max()}")
