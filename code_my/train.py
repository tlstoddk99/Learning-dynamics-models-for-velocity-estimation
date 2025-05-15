import os
import time
import argparse
from copy import deepcopy
import pandas as pd
import wandb
import matplotlib.pyplot as plt

import torch
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

# Local imports
from sensor_models.sensor_refine_model import TCNGaussian
from sensor_models.sensor_dataset import SensorDataset
from utils.argparser import get_parser




def parse_args():
    parser = get_parser()
    return parser.parse_args()


def build_model(input_size: int, output_size: int, args) -> torch.nn.Module:
    model = TCNGaussian(
        input_size=4,
        output_size=4,
        num_channels=256,
        num_levels=4,
        kernel_size=2,
        dropout=0.2,
        # activation=torch.nn.ReLU,
        activation=torch.nn.SiLU,
        eps=1e-3
    )
    return model.to(args.device)


def get_dataloader(df, args, train: bool) -> DataLoader:
    dataset = SensorDataset(
        df,
        subsample_all=args.common_downsample_all,
        Ts_multiplier=args.common_Ts_mult,
        check_new_run=True,
        test_run_id=args.common_test_run_id,
        test=not train,
        dtype=torch.float32 if args.common_precision == 32 else torch.float64,
        device=args.device,
        dataset_scaler=args.common_dataset_scaler,
        sequence_length=200,
    )
    return DataLoader(
        dataset,
        batch_size=args.ukf_batch_size,
        shuffle=train,
        # num_workers=args.common_loader_workers,
    )


def calculate_loss(model, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    mu, var = model(inputs)
    loss = torch.functional.F.gaussian_nll_loss(
        input=mu,
        target=targets,
        var=var,
        full=True,
        reduction='mean'
    )
    return loss


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    losses = []
    for batch in dataloader:
        imu = batch[:, 7:10,:]
        wheel = batch[:, 3:4,:]
        
        inputs = torch.cat((imu, wheel), dim=1).to(device)
        targets = batch[:, -4:, -1].to(device)

        optimizer.zero_grad()
        loss = calculate_loss(model, inputs, targets)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    return np.mean(losses)


def validate(model, dataloader, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in dataloader:
            imu = batch[:,7:10,:]
            wheel = batch[:,3:4,:]
            inputs = torch.cat((imu, wheel), dim=1).to(device)
            targets = batch[:, -4:, -1].to(device)

            loss = calculate_loss(model, inputs, targets)
            losses.append(loss.item())
    return np.mean(losses)


def main():
    args = parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Paths and logging setup
    df = pd.read_csv('/home/a/Learning-dynamics-models-for-velocity-estimation/code_my/opti_test/hoons_all_train_and_val.csv', index_col=0)
    timestamp = time.strftime('%m-%d_%H-%M')
    save_dir = os.path.join('/home/a/Learning-dynamics-models-for-velocity-estimation/code_my/trained_models/', timestamp)
    os.makedirs(save_dir, exist_ok=True)

    # Build model, dataloaders, optimizer
    model = build_model(input_size=4, output_size=4, args=args)
    train_loader = get_dataloader(df, args, train=True)
    val_loader = get_dataloader(df, args, train=False)
    optimizer = Adam(model.parameters(), lr=args.ukf_lr)

    best_val_loss = float('inf')
    train_loss_history = []
    val_loss_history = []
    for epoch in range(1, 5000):
        train_loss = train_one_epoch(model, train_loader, optimizer, args.device)
        val_loss = validate(model, val_loader, args.device)

        
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"Epoch {epoch:04d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            save_path = os.path.join(save_dir, f"best_epoch_{epoch}_loss_{val_loss:.4f}.pt")
            torch.save(model.state_dict(), save_path)

    print("Training complete. Best validation loss: {:.4f}".format(best_val_loss))
    
    # Plotting
    plt.figure()
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over epochs')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_plot.png'))
    plt.close()
    plt.show()


if __name__ == '__main__':
    main()
