import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

class VehicleDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df
        