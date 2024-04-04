import torch
import numpy as np
from torch.utils.data import Dataset


class PointsDataset(Dataset):
    def __init__(self, path_to_txt_file):
        self.data = self.load_data(path_to_txt_file)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    @staticmethod
    def load_data(path: str):
        points = []
        with open(path, "r") as f:
            line = f.readline()
            while line:
                x, y = map(float, line.split(" "))
                points.append((x, y))
                line = f.readline()
        return torch.tensor(points, dtype=torch.float)
