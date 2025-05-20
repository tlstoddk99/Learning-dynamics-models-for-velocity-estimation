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
from sensor_models.debias_model import TCNGaussian
from sensor_models.debias_mlp_model import MLPGaussian
from sensor_models.de_bias_dataset import DeBiasDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# fix seed
torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)




def build_model(input_size: int, output_size: int) -> torch.nn.Module:
    model = TCNGaussian()      
    return model.to(device)


def get_dataloader(df, save_path):
    # Prepare train/validation datasets
    train_ids, val_ids = [10, 12, 23, 27, 28, 29, 31, 32], [0, 2, 6, 7, 13, 18]
    train_ds = DeBiasDataset(df, run_ids=train_ids, device=device)
    val_ds   = DeBiasDataset(df, run_ids=val_ids,   device=device)

    # Build DataLoaders
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False)


    return train_loader, val_loader
  
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    losses = []
    for inputs,targets in dataloader:
        optimizer.zero_grad()
        mu, var = model(inputs)
        loss = model.loss_function(mu, var, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
    return np.mean(losses)


def validate(model, dataloader, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            mu, var = model(inputs)
            loss = model.loss_function(mu, var, targets)
            losses.append(loss.item())
            
    return np.mean(losses)


def main():
    # Paths and logging setup
    df = pd.read_csv('/home/a/Learning-dynamics-models-for-velocity-estimation/code_my/opti_test/hoons_all_train_and_val.csv', index_col=0)
    timestamp = time.strftime('%m-%d_%H-%M')
    save_dir = os.path.join('/home/a/Learning-dynamics-models-for-velocity-estimation/code_my/trained_models/', timestamp)
    os.makedirs(save_dir, exist_ok=True)

    # Build model, dataloaders, optimizer
    model = build_model(input_size=3, output_size=3)
    train_loader, val_loader = get_dataloader(df, save_dir)
    optimizer = AdamW(model.parameters(), lr=1e-4)

    best_val_loss = float('inf')
    train_loss_history = []
    val_loss_history = []
    epoch_history = []
    MAX_LOSS = 10
    train_loss = 0.0
    val_loss = 0.0

    # start index: 1, end index: 1000
    for epoch in tqdm(range(1, 1001), unit="epoch"):
        train_loss = train_one_epoch(model, train_loader, optimizer, device=device)
        val_loss = validate(model, val_loader, device=device)

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
