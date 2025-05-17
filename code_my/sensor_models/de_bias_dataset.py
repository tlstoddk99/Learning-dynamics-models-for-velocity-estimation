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
        
        self.plot = plot

        # Fill missing values
        df = self._fill_na(df)
        
        # self.raw_imu = df[["ax_imu", "ay_imu", "r_imu"]].to_numpy()
        # self.gt_imu = self.make_gt_imu(df)
        # optimal_taus, optimal_ms=self.optimal_window_size(
        #     self.raw_imu, self.gt_imu, self.dt
        # )
        # print(f"df length: {len(df)}")
        # print(f"Optimal averaging time: {optimal_taus}, Window size: {optimal_ms}")
        
        
        
        # Split DataFrame by run
        runs = self._split_runs(df)
        # If plotting is enabled and run is long enough, visualize IMU data
        for run_id, run_df in runs.items():
            if plot and len(run_df) > self.input_seq_len + self.pred_seq_len:    
                self.plot_imu(run_df)
        
        # Generate inputs and targets
        inputs, targets = self._generate_samples(runs)

        # Convert to torch tensors
        self.inputs = torch.tensor(
            np.stack(inputs), dtype=self.dtype, device=self.device
        )  # [N, input_seq_len, 4]
        self.targets = torch.tensor(
            np.stack(targets), dtype=self.dtype, device=self.device
        )  # [N, 4]
        

    def _fill_na(self, df: pd.DataFrame) -> pd.DataFrame:
        # Replace NaN values with zero
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
        Create samples using a sliding window for a single run.
        """
        inputs, targets = [], []
        total_len = len(run_df)
        step = self.pred_seq_len
        # Slide window with stride equal to prediction length
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
        # cols = ["ax_imu", "ay_imu", "r_imu", "omega_wheels"]
        # return df.loc[start : start + self.input_seq_len - 1, cols].values
        cols = ["ax_imu", "ay_imu", "r_imu"]
        input_data=df.loc[start : start + self.input_seq_len - 1, cols].values
        return input_data.T

    def _get_target_window(
        self, df: pd.DataFrame, start: int
    ) -> list:
        """
        Calculate mean IMU biases and friction coefficient over prediction window.
        """
        pred = df.loc[
            start + self.input_seq_len :
            start + self.input_seq_len + self.pred_seq_len - 1
        ]

        # Compute average IMU biases
        mean_ax, mean_ay, mean_r = self._compute_mean_biases(pred)

        # Compute average friction coefficient
        # friction_mean = pred["friction"].mean()

        # return [mean_ax, mean_ay, mean_r, friction_mean]
        return [mean_ax, mean_ay, mean_r]

    def _compute_mean_biases(self, pred: pd.DataFrame) -> tuple:
        """
        Compute average difference between true accelerations/angular rate
        and IMU measurements over prediction window.
        """
        # True accelerations from motion capture data
        dvx = pred["v_x"].diff().fillna(0.0) / self.dt
        dvy = pred["v_y"].diff().fillna(0.0) / self.dt
        psi = pred["r"]

        ax_true = dvx - psi * pred["v_y"]
        ay_true = dvy + psi * pred["v_x"]
        r_true = pred["r"]

        # Bias = IMU reading - true value
        ax_bias = pred["ax_imu"] - ax_true
        ay_bias = pred["ay_imu"] - ay_true
        r_bias = pred["r_imu"] - r_true
        return ax_bias.mean(), ay_bias.mean(), r_bias.mean()

    def plot_imu(self, run_df):
        """
        Plot IMU sensor data and ground truth biases for each segment.
        """
        n = len(run_df)
        time = np.arange(n) * self.dt

        # IMU measurements as numpy arrays
        ax_imu = run_df["ax_imu"].to_numpy()
        ay_imu = run_df["ay_imu"].to_numpy()
        r_imu  = run_df["r_imu"].to_numpy()

        # Compute ground truth accelerations/turn rate
        dvx = run_df["v_x"].diff().fillna(0.0).to_numpy() / self.dt
        dvy = run_df["v_y"].diff().fillna(0.0).to_numpy() / self.dt
        psi = run_df["r"].to_numpy()
        ax_true = dvx - psi * run_df["v_y"].to_numpy()
        ay_true = dvy + psi * run_df["v_x"].to_numpy()
        r_true  = run_df["r"].to_numpy()
        
        
        raw_imu = np.column_stack((ax_imu, ay_imu, r_imu))
        gt_imu = np.column_stack((ax_true, ay_true, r_true))
        
        
        optimal_tau, optimal_m = self.optimal_window_size(raw_imu, gt_imu, self.dt)
        print(f"run_id: {run_df['run_id'].iloc[0]}, run_length: {n}")
        # print(f"Optimal averaging time: {optimal_tau:.2f} s, Window size: {optimal_m} samples")
        print(f"Optimal averaging time: {optimal_tau}, Window size: {optimal_m}")

        # Prepare arrays for biases
        pred = self.pred_seq_len
        ax_bias = np.zeros(n)
        ay_bias = np.zeros(n)
        r_bias  = np.zeros(n)

        # Fill segment-wise average biases
        for start in range(0, n, pred):
            end = min(start + pred, n)
            ax_b = run_df["ax_imu"].iloc[start:end].mean() - ax_true[start:end].mean()
            ay_b = run_df["ay_imu"].iloc[start:end].mean() - ay_true[start:end].mean()
            r_b  = run_df["r_imu"].iloc[start:end].mean()  - r_true[start:end].mean()
            ax_bias[start:end] = ax_b
            ay_bias[start:end] = ay_b
            r_bias[start:end]   = r_b

        # Debiased signals
        debiased_ax = ax_imu - ax_bias
        debiased_ay = ay_imu - ay_bias
        debiased_r  = r_imu  - r_bias

        # Plot biases
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
        dvx = df["v_x"].diff().fillna(0.0) / self.dt
        dvy = df["v_y"].diff().fillna(0.0) / self.dt
        psi = df["r"].to_numpy()
        ax_true = dvx - psi * df["v_y"].to_numpy()
        ay_true = dvy + psi * df["v_x"].to_numpy()
        r_true  = df["r"].to_numpy()
        return np.column_stack((ax_true, ay_true, r_true))
    
    def allan_deviation(self,x: np.ndarray, dt: float):
        """
        Compute Allan deviation for given 1D signal x with sampling period dt.
        Returns:
            taus (np.ndarray): Array of averaging times τ = m * dt
            adevs (np.ndarray): Corresponding Allan deviations σ(τ)
            ms (np.ndarray): Window sizes m (in samples)
        """
        N = len(x)
        max_m = N // 2
        # Use powers of two for window sizes
        exponent_max = int(np.floor(np.log2(max_m)))
        ms = 2 ** np.arange(0, exponent_max + 1)
        taus = ms * dt
        adevs = np.zeros_like(taus, dtype=float)
        
        for i, m in enumerate(ms):
            M = N // m
            # Compute segment means
            x_bar = np.array([np.mean(x[j*m:(j+1)*m]) for j in range(M)])
            # Compute Allan variance for this m
            diffs = np.diff(x_bar)
            adevs[i] = np.sqrt(0.5 * np.mean(diffs**2))
        
        return taus, adevs, ms

    def optimal_window_size(self, raw: np.ndarray, gt: np.ndarray, dt: float):
        """
        채널별 최적 윈도우 크기를 반환합니다.
        Returns:
            optimal_taus (np.ndarray): 채널별 최적 τ (초)
            optimal_ms  (np.ndarray): 채널별 최적 m (샘플 개수)
        """
        residual = raw - gt       # shape (N, 3)
        N, C = residual.shape     # C = 3
        optimal_taus = np.zeros(C)
        optimal_ms   = np.zeros(C, dtype=int)
        
        t = np.arange(N)
        for i in range(C):
            # 1) 채널 i의 신호만 뽑아서 추세 제거
            res_i = residual[:, i]
            p = np.polyfit(t, res_i, 1)
            trend = np.polyval(p, t)      # shape (N,)
            detrended = res_i - trend     # shape (N,)
            
            # 2) Allan 편차 계산
            taus, adevs, ms = self.allan_deviation(detrended, dt)
            
            # 3) 첫 포인트(τ=T0)는 제외하고 최솟값 인덱스 찾기
            idx_min = np.argmin(adevs[1:]) + 1
            optimal_taus[i] = taus[idx_min]
            optimal_ms[i]  = int(ms[idx_min])
        
        return optimal_taus, optimal_ms
    
    def __len__(self):
        # Return number of samples
        return self.inputs.size(0)

    def __getitem__(self, idx: int):
        # Get input sequence and target at index
        return self.inputs[idx], self.targets[idx]


if __name__ == "__main__":
    # Load dataset CSV
    # df = pd.read_csv("/home/a/Learning-dynamics-models-for-velocity-estimation/code/opti_test/hoons_all_train_and_val.csv")
    # For test set use:
    df = pd.read_csv("/home/a/Learning-dynamics-models-for-velocity-estimation/code/opti_test/hoons_all_test.csv")
    # Create dataset and plot if desired
    dataset = DeBiasDataset(df, plot=True)
