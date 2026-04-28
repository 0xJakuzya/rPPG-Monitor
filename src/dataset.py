import numpy as np
import torch
from torch.utils.data import Dataset


class RPPGDataset(Dataset):

    def __init__(self, files: list[str]):
        self.files = files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        npz = np.load(self.files[idx])
        rgb = torch.from_numpy(npz["rgb"])  
        ppg = torch.from_numpy(npz["ppg"])  
        return rgb, ppg
