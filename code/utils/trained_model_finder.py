import json
import pickle
from pathlib import Path
import torch


def find_model(args, mode, return_path=False):
    """
        Find model trained with the same configuration as the current run.
        If not found, raise FileNotFoundError
    """
    def check_model_config(args, model_args):
        checking_order = ["common", "base", "res", "ukf"]
        for stage in checking_order:
            for key, value in vars(args).items():
                if key.startswith(stage):
                    try:
                        if model_args[key] != value:
                            return False
                    except KeyError:
                        return False
            if stage == mode:
                break
        return True

    models_path = Path(__file__).parent.parent / "trained_models"
    print(f'model path  {models_path}')
    for model_path in models_path.iterdir():
        if model_path.is_dir() and model_path.name.startswith(mode):
            with open(model_path / "args.json", "r") as fh:
                model_args = json.load(fh)
            if check_model_config(args, model_args):

                model_dict = torch.load(model_path / "model.pt")

                if return_path:
                    return model_path.absolute()
                return model_dict

    raise FileNotFoundError(f"No {mode} model found")
