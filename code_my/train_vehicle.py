import os
import time
import pandas as pd
import wandb
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

# Local imports
from robot_models.vehicle_dataset import VehicleDataset
from robot_models.single_track_pacejka import SingleTrackPacejkaModel
# from robot_models.pacejka import PacejkaTireModel
from robot_models.neural_tire_model import NeuralPacejkaTireModel
from robot_models.single_track_parameters import SingleTrackParameters
import torchdiffeq as ode

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# fix seed
torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

state_weights = torch.tensor([0.2225, 0.5064, 0.1566, 0.1145, 0.0, 0.0, 0.0], device=device)

def get_dataloader(df):
    # Prepare train/validation datasets
    # df = preprocess_df(df)

    train_ids, val_ids = [10, 12, 23, 27, 28, 29, 31, 32], [0, 2, 6, 7, 13, 18]
    train_ds = VehicleDataset(df, run_ids=train_ids)
    val_ds   = VehicleDataset(df, run_ids=val_ids)

    # Build DataLoaders
    train_loader = DataLoader(train_ds, batch_size=2048, shuffle=False)
    val_loader   = DataLoader(val_ds,   batch_size=2048, shuffle=False)
    return train_loader, val_loader

def train_one_epoch(model, dataloader, optimizer, scheduler):
    model.train()
    losses = []
    for x, x_next in dataloader:
        x = x.to(device)
        x_next = x_next.to(device)
        x=x[:, :7]
        x_next=x_next[:, :7]
        optimizer.zero_grad()
        
        pred_x = ode.odeint(
            model,
            x,
            torch.tensor([0, 0.01], device=x.device),
            method="rk4",
            rtol=1e-5,
            atol=1e-6
        )[-1]
        
        loss = model.loss_function(pred_x, x_next).mul(state_weights).sum()
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
        
    return np.mean(losses)


def validate(model, dataloader):
    model.eval()
    losses = []
    with torch.no_grad():
        for x, x_next in dataloader:
            x = x.to(device)
            x_next = x_next.to(device)
            x=x[:, :7]
            x_next=x_next[:, :7]
            
            pred_x = ode.odeint(
                model,
                x,
                torch.tensor([0, 0.01], device=x.device),
                method="rk4",
                rtol=1e-5,
                atol=1e-6
            )[-1]
            
            loss = model.loss_function(pred_x, x_next).mul(state_weights).sum()
            losses.append(loss.item())
            
    return np.mean(losses)


def main():
    # Paths and logging setup
    df = pd.read_csv('/home/a/Learning-dynamics-models-for-velocity-estimation/code_my/dataset/hoons_all_train_and_val_gt.csv', index_col=0)
    timestamp = time.strftime('%m-%d_%H-%M')
    save_dir = os.path.join('/home/a/Learning-dynamics-models-for-velocity-estimation/code_my/trained_models/vehicle', timestamp)
    os.makedirs(save_dir, exist_ok=True)

    total_epochs = 200
    vehicle_params = SingleTrackParameters()
    # tire_model = PacejkaTireModel(vehicle_parameters=vehicle_params)
    tire_model = NeuralPacejkaTireModel(vehicle_parameters=vehicle_params)
    model= SingleTrackPacejkaModel(
                                    vehicle_parameters=vehicle_params,
                                    tire_model=tire_model)
    model.to(device)
    train_loader, val_loader = get_dataloader(df)
    
    # 1) Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=0.0005, 
        weight_decay=0
    )

    # 2) Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=0.0005, 
        steps_per_epoch=len(train_loader),
        epochs=total_epochs
    )

    best_val_loss = float('inf')
    train_loss_history = []
    val_loss_history = []
    epoch_history = []
    MAX_LOSS = 100
    train_loss = 0.0
    val_loss = 0.0

    # start index: 1, end index: 400
    for epoch in tqdm(range(1, total_epochs + 1), unit="epoch"):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler)
        val_loss = validate(model, val_loader)

        train_loss_history.append(np.clip(train_loss, -MAX_LOSS, MAX_LOSS))
        val_loss_history.append(np.clip(val_loss,   -MAX_LOSS, MAX_LOSS))
        epoch_history.append(epoch)
    
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"\nTrain Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            save_path = os.path.join(save_dir, f"best_epoch_{epoch}_loss_{val_loss:.4f}.pt")
            torch.save(model.state_dict(), save_path)

    print("Training complete. Best validation loss: {:.4f}".format(best_val_loss))
    
    
    plt.figure(dpi=500)
    plt.plot(epoch_history, train_loss_history, label='Train Loss')
    plt.plot(epoch_history, val_loss_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over epochs')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_plot.png'),dpi=500, bbox_inches='tight')
    plt.close()



if __name__ == '__main__':
    main()
