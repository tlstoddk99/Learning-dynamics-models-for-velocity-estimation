import torch
import torch.nn as nn
from noise_models.crosscovariance_noise_model import CrossCovNoiseModel
from noise_models.diagonal_noise_model import ConstDiagonalNoiseModel
from noise_models.heterogenus_noise_model import HeteroCovNoiseModel


def get_noise_model(args, state_dim=5, meas_dim=4):
    device = torch.device(args.common_device)
    if args.ukf_noise_model == "crosscov":
        return CrossCovNoiseModel(state_dim, meas_dim, device=device)
    elif args.ukf_noise_model == "diagonal":
        return ConstDiagonalNoiseModel(state_dim, meas_dim, device=device)
    elif args.ukf_noise_model == "hetero":
        return HeteroCovNoiseModel(state_dim, meas_dim, device=device)
    else:
        raise ValueError("Unknown noise model")
