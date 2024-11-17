import os
import json
import torch
from pathlib import Path
import wandb
from utils.trained_model_finder import find_model


def model_saver(args, best_model, datetime, epoch=0):
    save_path = (Path(__file__).parent.parent /
                 "trained_models" / f"ukf_{datetime}")
    os.makedirs(save_path, exist_ok=True)

    if epoch == 0:
        with open(os.path.join(save_path, "args.json"), "w") as fh:
            json.dump(vars(args), fh, indent=4)

    model_state_dict = best_model[0]
    noise_model_state_dict = best_model[1]

    model_filename = f'model_at_{epoch}.pt'

    torch.save(model_state_dict, os.path.join(
        save_path, model_filename))

    noise_model_filename = f'noise_model_at_{epoch}.pt'
    torch.save(noise_model_state_dict, os.path.join(
        save_path, noise_model_filename))

    artefact = wandb.Artifact(f"ukf_{datetime}_at_{epoch}", type="model")
    artefact.add_file(os.path.join(save_path, "args.json"))
    artefact.add_file(os.path.join(save_path, model_filename))
    artefact.add_file(os.path.join(save_path, noise_model_filename))
    return artefact
