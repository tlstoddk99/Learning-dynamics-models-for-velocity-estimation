from copy import deepcopy
from distutils import config
import os
import pickle
import torch
import numpy as np
import pandas as pd
import wandb
import time

import torchdiffeq as ode
import matplotlib.pyplot as plt
from pathlib import Path
from datasets.optitrack import OptitrackDataset
from preprocessors.no import NoPreprocessing
from preprocessors.orientation import OrientationPreprocessing
from preprocessors.imu_offset import ImuOffestAndRotation
from robot_models.residual_model import ResidualNeuralModel

from robot_models.single_track_pacejka import SingleTrackPacejkaModel, observation
from tire_models.pacejka import PacejkaTireModel
from utils.state_wrapper import STATE_DEF_LIST
from utils.argparser import get_parser

from utils.trained_model_finder import find_model
from datasets.dataset_getter import get_loaders, get_sequence_loaders
from utils.solver_settings import get_solver_settings, get_ode_solve_method
from utils.base_model_training import train_base_model
from utils.res_model_training import train_residual_model
from utils.ukf_model_saver import model_saver

# ukf
from datasets.optitrack_sequential import OptitrackDatasetSequential
from noise_models.diagonal_noise_model import ConstDiagonalNoiseModel
from noise_models.crosscovariance_noise_model import CrossCovNoiseModel
from noise_models.heterogenus_noise_model import HeteroCovNoiseModel
from filters.ukf_model_steper_inference import UKFModelStepperInference
from filters.ukf_model_steper_training import UKFModelStepperTrain
from utils.dataframe_timeseries_save import save_as_dataframe
from utils.res_model_training import create_residual_model
import json
from utils.wanb_model_getter import get_wandb_model


torch.set_grad_enabled(False)


model_dict = {
    # 'model': '<wandb_path>',
}

pred_model_name = model_dict['model']

args, pred_model, pred_noise_model = get_wandb_model(pred_model_name)


Q, R, P = pred_noise_model(torch.zeros(1, 5))
print(f'pred noise model')
print(f"Q: {torch.diag(Q[0])}")
print(f"R: {torch.diag(R[0])}")
print(f"P: {torch.diag(P[0])}")


device = torch.device(args.common_device)
state_weights = torch.tensor(args.ukf_states_weights, device=device)

loss_fn = torch.nn.MSELoss(reduction="none")
dataset_path = Path("opti_test/hoons_all_test.csv")

test_dataset = OptitrackDatasetSequential(
    csv_file=dataset_path,
    subsample_all=args.common_downsample_all,
    Ts_multiplier=args.common_Ts_mult,
    check_new_run=True,
    test_run_id=[],
    test=False,
    dtype=torch.float32 if args.common_precision == 32 else torch.float64,
    device=torch.device(args.common_device),
    dataset_scaler=1.0,
    sequence_length=args.ukf_test_sequence_length,
)

test_data_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.ukf_batch_size, shuffle=False, num_workers=args.common_loader_workers
)

state_dim = 5
control_dim = 2

solver_settings = get_solver_settings(args)

ukf_single_stepper = UKFModelStepperInference(pred_model, pred_noise_model,
                                              test_dataset.dt, solver_settings,
                                              q_entr_lb=args.ukf_q_entropy_lb,
                                              device=device)


ukf_stepper = UKFModelStepperTrain(ukf_model_steper=ukf_single_stepper)

args.ukf_start_loss = 200
state_weights[-2:] = 0.0
print(f"state weights: {state_weights}")

loss_list = []


def calc_loss(x, args, test=False):
    X0 = x[:, 0, :state_dim]
    # print(f"X0: {X0}")
    # X0[:, -1] = 0.4
    u = x[:, :, state_dim: state_dim + control_dim]
    imu = x[:, :, -3:]
    wheel_speed = x[:, :, 3].unsqueeze(-1)
    y = torch.cat((imu, wheel_speed), dim=-1)
    X_ukf, P, q_entropy, r_entropy = ukf_stepper(X0, u, y)

    e = X_ukf[:, :, :state_dim] - x[:, :, :state_dim]
    print(f"e: {e.abs().sum()}")

    loss_each = loss_fn(X_ukf[:, args.ukf_start_loss:, :],
                        x[:, args.ukf_start_loss:, :state_dim]).mul(state_weights)
    loss_list.append(loss_each)
    loss = loss_each.sum()
    return loss, q_entropy, r_entropy, X_ukf, x[:, :, :state_dim], P


test_loss = 0.0
with torch.no_grad():
    df = None
    for x in test_data_loader:
        loss, q_entropy, r_entropy, X_ukf, x, P = calc_loss(x, args, test=True)
        test_loss += loss.item()
        if df is None:
            df = save_as_dataframe(X_ukf, x, P)


test_loss = test_loss / \
    len(test_dataset) / (args.ukf_test_sequence_length - args.ukf_start_loss)
print(f"test loss: {test_loss}")


errors_df = df[['vx_e', 'vy_e', 'r_e', 'omega_e_e']]
state_weights = {'vx_e': 0.2225, 'vy_e': 0.5064,
                 'r_e': 0.1566, 'omega_e_e': 0.1145}
error = (errors_df.abs() * state_weights).sum(axis=1)

# print mse for all columns combined
loss_list = torch.cat(loss_list, dim=1).sum(dim=-1)
print(f"MSE all: {loss_list.mean()}")
np.save("pred_ukf_obs_sp.npy", loss_list.cpu().numpy())

for col in errors_df.columns:
    # print mse
    mse = (errors_df[col]**2).mean() * state_weights[col]
    print(f'MSE {col}: {mse:.6f}')

print(f'{errors_df.describe()}')

print(f'Error  mean: {error.mean():.6f}')
print(f'Error  median: {error.median():.6f}')
print(f'99 precentile: {error.quantile(0.99):.6f}')

time = df['time'].values
t_data = time

confidence = 1

# sqrt all columns with 'P' in the name
for col in df.columns:
    if 'P' in col:
        df[col] = np.sqrt(df[col])

plt.figure()
plt.plot(t_data, df['vx_ukf'].values, label="vx est")
plt.plot(t_data, df['vx_gt'].values, label="vx")
plt.fill_between(
    t_data,
    df['vx_ukf'].values - df['P_vx'].values * confidence,
    df['vx_ukf'].values + df['P_vx'].values * confidence,
    alpha=0.5,
)
plt.legend()


plt.figure()
plt.subplot(2, 1, 1)
plt.plot(t_data, df['vy_ukf'].values, label="vy est")
plt.plot(t_data, df['vy_gt'].values, label="vy")
plt.fill_between(
    t_data,
    df['vy_ukf'].values - df['P_vy'].values * confidence,
    df['vy_ukf'].values + df['P_vy'].values * confidence,
    alpha=0.5,
)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t_data, df['vy_e'].values, label="vy_e")


plt.figure()
plt.plot(t_data, df['r_ukf'].values, label="r est")
plt.plot(t_data, df['r_gt'].values, label="r")
plt.fill_between(
    t_data,
    df['r_ukf'].values - df['P_r'].values * confidence,
    df['r_ukf'].values + df['P_r'].values * confidence,
    alpha=0.5,
)
plt.legend()

plt.figure()
plt.plot(t_data, df['friction_ukf'].values, label="friction est")
plt.plot(t_data, df['friction_gt'].values, label="friction")
plt.fill_between(
    t_data,
    df['friction_ukf'].values - df['P_friction'].values * confidence,
    df['friction_ukf'].values + df['P_friction'].values * confidence,
    alpha=0.5,
)
plt.legend()

plt.show()
