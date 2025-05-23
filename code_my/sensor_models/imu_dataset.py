import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

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
    lpf_cutoff: float = 2,
    lpf_order: int = 4,
) -> pd.DataFrame:
    """
    1) Fill NaNs
    2) Compute IMU ground truth from motion-capture velocities
    3) Compute error signals (error = noise + bias)
    4) Separate noise and bias by applying a low-pass filter
    5) Compute noise and bias signals
    
    """
  
    df = df.fillna(0)

    nyquist = 0.5 / dt
    norm_cut = lpf_cutoff / nyquist
    b, a = butter(lpf_order, norm_cut, btype="low", analog=False)

    # Compute GT IMU
    dvx = df["v_x"].diff().fillna(0.0) / dt
    dvy = df["v_y"].diff().fillna(0.0) / dt
    yaw_rate = df["r"]
    df["ax_gt"] = dvx - yaw_rate * df["v_y"]
    df["ay_gt"] = dvy + yaw_rate * df["v_x"]
    df["r_gt"] = yaw_rate
    
    # Normalize IMU signals
    df["ax_imu"], df["ay_imu"], df["r_imu"] = normalize_imu(
        df["ax_imu"].to_numpy(),
        df["ay_imu"].to_numpy(),
        df["r_imu"].to_numpy()
    )
    # Normalize GT signals
    df["ax_gt"], df["ay_gt"], df["r_gt"] = normalize_imu(
        df["ax_gt"].to_numpy(),
        df["ay_gt"].to_numpy(),
        df["r_gt"].to_numpy()
    )

    # diff = noise + bias
    df["ax_e"] = df["ax_imu"] - df["ax_gt"]
    df["ay_e"] = df["ay_imu"] - df["ay_gt"]
    df["r_e"] = df["r_imu"] - df["r_gt"]

    # separate noise and bias by applying LPF
    for col in ["ax_e", "ay_e", "r_e"]:
        df[f"{col}_b"] = filtfilt(b, a, df[col].to_numpy())

    # noise = error - bias
    df["ax_n"] = df["ax_e"] - df["ax_e_b"]
    df["ay_n"] = df["ay_e"] - df["ay_e_b"]
    df["r_n"] = df["r_e"] - df["r_e_b"]

    return df


class IMUDataset(torch.utils.data.Dataset):
    """
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
        step: int = 100,
        pad: int = 50,
        run_ids: list = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.input_seq_len = input_seq_len
        self.pred_seq_len = pred_seq_len
        self.step = step
        self.pad = pad
        self.device = device

        # Split runs
        self.runs = self._split_and_trim_runs(df, run_ids)

        # Generate samples
        inputs, targets = self._generate_samples()
        self.inputs = torch.tensor(np.stack(inputs), dtype=torch.float32, device=device)
        self.targets = torch.tensor(np.stack(targets), dtype=torch.float32, device=device)

    def _split_and_trim_runs(self, df: pd.DataFrame, run_ids: list = None) -> dict:
        raw_runs = {}
        ids = run_ids if run_ids is not None else df["run_id"].unique()
        for rid in ids:
            run_df = df[df["run_id"] == rid].reset_index(drop=True)
            raw_runs[rid] = run_df

        processed = {}
        min_len = self.input_seq_len + self.pred_seq_len + 2 * self.pad
        for rid, run_df in raw_runs.items():
            if len(run_df) >= min_len:
                # drop pad frames at start/end
                processed[rid] = run_df.iloc[self.pad : -self.pad].reset_index(drop=True)
        return processed

    def _generate_samples(self) -> tuple:
        all_inputs, all_targets = [], []
        for run_df in self.runs.values():
            ins, tars = self._process_single_run(run_df)
            all_inputs.extend(ins)
            all_targets.extend(tars)
        return all_inputs, all_targets

    def _process_single_run(self, run_df: pd.DataFrame) -> tuple:
        inputs, targets = [], []
        total = len(run_df)
        for start in range(0, total - self.input_seq_len - self.pred_seq_len + 1, self.step):
            inp = self._get_input_window(run_df, start)
            tar = self._get_target_window(run_df, start)
            inputs.append(inp)
            targets.append(tar)
        return inputs, targets

    def _get_input_window(self, df: pd.DataFrame, start: int) -> np.ndarray:
        cols = ["ax_imu", "ay_imu", "r_imu"]
        data = df.loc[start : start + self.input_seq_len - 1, cols].values
        # return data.T
        return data

    def _get_target_window(self, df: pd.DataFrame, start: int) -> np.ndarray:
        idx = start + self.input_seq_len + self.pred_seq_len - 1
        # cols = ["ax_gt", "ay_gt", "r_gt",
        #         "ax_e_b", "ay_e_b", "r_e_b",
        #         "ax_n", "ay_n", "r_n"]
        cols = ["ax_e_b", "ay_e_b", "r_e_b"]
        # return df.loc[idx, cols].to_numpy()
        # data = df.loc[start + self.input_seq_len : start + self.input_seq_len + self.pred_seq_len - 1, cols].values
        data= df.loc[idx, cols].values
        return data

    def __len__(self) -> int:
        return self.inputs.size(0)

    def __getitem__(self, idx: int) -> tuple:
        return self.inputs[idx], self.targets[idx]

    def plot(self, run_id: int, save_path: str = None):
        """
        Plot filtered error signals for a specific run.
        If save_path is provided, saves the figure instead of showing.
        """
        run_df = self.runs[run_id]
        n = len(run_df)
        time = np.arange(n) * 0.01
        
        ax_imu = run_df["ax_imu"].to_numpy()
        ay_imu = run_df["ay_imu"].to_numpy()
        r_imu = run_df["r_imu"].to_numpy()
        
        ax_gt = run_df["ax_gt"].to_numpy()
        ax_b = run_df["ax_e_b"].to_numpy()
        ax_n = run_df["ax_n"].to_numpy()

        dyn_ay = run_df["ay_gt"].to_numpy()
        ay_b = run_df["ay_e_b"].to_numpy()
        ay_n = run_df["ay_n"].to_numpy()

        dyn_r = run_df["r_gt"].to_numpy()
        r_b = run_df["r_e_b"].to_numpy()
        r_n = run_df["r_n"].to_numpy()

        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 6))
        fig.suptitle(f"Run ID: {run_id}")
        axs[0].plot(time, ax_n, label="noise ax")
        axs[0].plot(time, ax_b, label="bias ax")

        axs[0].set_ylabel("ax")
        axs[1].plot(time, ay_n, label="noise ay")
        axs[1].plot(time, ay_b, label="bias ay")

        axs[1].set_ylabel("ay")
        axs[2].plot(time, r_n, label="noise r")
        axs[2].plot(time, r_b, label="bias r")

        axs[2].set_ylabel("r")
        axs[2].set_xlabel("Time (s)")
        for ax in axs:
            ax.legend()
        plt.tight_layout()
        
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 6))
        fig.suptitle(f"Run ID: {run_id}")
        axs[0].plot(time, ax_imu, label="ax imu")
        axs[0].plot(time, ax_gt, label="gt ax")
        axs[0].set_ylabel("ax imu")
        axs[1].plot(time, ay_imu, label="ay imu")
        axs[1].plot(time, dyn_ay, label="gt ay")
        axs[1].set_ylabel("ay imu")
        axs[2].plot(time, r_imu, label="r imu")
        axs[2].plot(time, dyn_r, label="gt r")
        axs[2].set_ylabel("r imu")
        axs[2].set_xlabel("Time (s)")
        for ax in axs:
            ax.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

            
# Example usage
if __name__ == "__main__":
    # Load your DataFrame here
    df = pd.read_csv("/home/a/Learning-dynamics-models-for-velocity-estimation/code_my/dataset/hoons_all_train_and_val.csv")
    # #normalize IMU signals
    # df["ax_imu"], df["ay_imu"], df["r_imu"] = normalize_imu(
    #     df["ax_imu"].to_numpy(),
    #     df["ay_imu"].to_numpy(),
    #     df["r_imu"].to_numpy()
    # )
    
    # Preprocess the DataFrame
    df = preprocess_df(df)
    

    # Create the dataset
    dataset = IMUDataset(df)

    for run_id in dataset.runs.keys():
        dataset.plot(run_id)
    
    # print(f"Dataset size: {len(dataset)}")
    # print(f"Input shape: {dataset.inputs.shape}")
    # print(f"Target shape: {dataset.targets.shape}")
    
    
    # print(f"ax_imu min/max: {dataset.inputs[:, 0, :].min()}, {dataset.inputs[:, 0, :].max()}")
    # print(f"ay_imu min/max: {dataset.inputs[:, 1, :].min()}, {dataset.inputs[:, 1, :].max()}")
    # print(f"r_imu min/max: {dataset.inputs[:, 2, :].min()}, {dataset.inputs[:, 2, :].max()}")
    # print(f"ax_bias min/max: {dataset.targets[:, 0].min()}, {dataset.targets[:, 0].max()}")
    # print(f"ay_bias min/max: {dataset.targets[:, 1].min()}, {dataset.targets[:, 1].max()}")
    # print(f"r_bias min/max: {dataset.targets[:, 2].min()}, {dataset.targets[:, 2].max()}")