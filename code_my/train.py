import torch
import pandas as pd
import numpy as np
import wandb
import time
# import sensor_models.sensor_refine_model
from sensor_models.sensor_refine_model import SensorRefineModel
from sensor_models.sensor_dataset import SensorDataset

from utils.argparser import get_parser

df=pd.read_csv('/home/a/Learning-dynamics-models-for-velocity-estimation/code_my/opti_test/hoons_all_train_and_val.csv', index_col=0)

args= get_parser().parse_args()

model=SensorRefineModel(
    input_size=4,
    output_size=4,
    hidden_size=args.ukf_hidden_size,
    num_layers=args.ukf_num_layers,
    dropout=args.ukf_dropout,
    batch_size=args.ukf_batch_size,
    sequence_length=args.ukf_sequence_length,
    device=torch.device(args.common_device),
    dtype=torch.float32 if args.common_precision == 32 else torch.float64,
    noise_model=args.ukf_noise_model,
    noise_model_kwargs={
        "input_size": 4,
        "hidden_size": args.ukf_hidden_size,
        "num_layers": args.ukf_num_layers,
        "dropout": args.ukf_dropout,
        "batch_size": args.ukf_batch_size,
        "sequence_length": args.ukf_sequence_length,
        "device": torch.device(args.common_device),
        "dtype": torch.float32 if args.common_precision == 32 else torch.float64
    }
)
model.to(args.common_device)


train_dataset = SensorDataset(
    df,
    subsample_all=args.common_downsample_all,
    Ts_multiplier=args.common_Ts_mult,
    check_new_run=True,
    test_run_id=args.common_test_run_id,
    test=False,
    dtype=torch.float32 if args.common_precision == 32 else torch.float64,
    device=torch.device(args.common_device),
    dataset_scaler=args.common_dataset_scaler,
    sequence_length=args.ukf_sequence_length
)

test_dataset = SensorDataset(
    df,
    subsample_all=args.common_downsample_all,
    Ts_multiplier=args.common_Ts_mult,
    check_new_run=True,
    test_run_id=args.common_test_run_id,
    test=True,
    dtype=torch.float32 if args.common_precision == 32 else torch.float64,
    device=torch.device(args.common_device),
    sequence_length=args.ukf_test_sequence_length
)
    

train_data_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.ukf_batch_size, shuffle=True, num_workers=args.common_loader_workers
)

test_data_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.ukf_batch_size, shuffle=False, num_workers=args.common_loader_workers
)


def calculate_loss(model, x):
    loss=torch.nn.GaussianNLLLoss(reduction='mean')
    
    return loss(
    

for epoch in range(args.ukf_epochs):
    for x in train_data_loader:
        imu=x[:,:,7:10]
        wheel=x[:,:,3]
        x_in=torch.cat((imu,wheel),dim=-1)
        gt=x[:,:,-4:]
        x_in=x_in.to(args.common_device)
        gt=gt.to(args.common_device)
        
        
       

                   
