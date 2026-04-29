import random
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from src import config

@dataclass(frozen=True)
class DatasetSplit:
    train_indices: list[int]
    val_indices: list[int]
    train_patients: set[str]
    val_patients: set[str]

def discover_window_files(data_dir: str | Path) -> list[str]:
    files = sorted(str(path) for path in Path(data_dir).rglob("*.npz"))
    if not files:
        print(f'No .npz windows found in {data_dir}.')
    return files

def describe_dataset(files: list[str]) -> None:
    sample = np.load(files[0])
    patch_shape = sample["patches"].shape
    ppg_shape = sample["ppg"].shape
    patients = {get_patient_id(file) for file in files}

    print(f"windows: {len(files)}")
    print(f"patients: {len(patients)}")
    print(f"sample patch shape: {patch_shape}")
    print(f"sample ppg shape: {ppg_shape}")

    if len(patch_shape) != 5 or patch_shape[-1] != 3:
        raise ValueError(f"Expected patch shape [time, roi, h, w, 3], got {patch_shape}")
    if patch_shape[1] != config.MULTI_ROI_COUNT:
        raise ValueError(f"Expected {config.MULTI_ROI_COUNT} ROI, got {patch_shape[1]}")
    if patch_shape[2] != config.ROI_PATCH_SIZE or patch_shape[3] != config.ROI_PATCH_SIZE:
        raise ValueError(
            f"Expected patch size {config.ROI_PATCH_SIZE}, got {(patch_shape[2], patch_shape[3])}"
        )


def split_by_patient(files: list[str], val_split: float, seed: int) -> DatasetSplit:
    patient_to_indices: dict[str, list[int]] = {}
    for index, file in enumerate(files):
        patient_to_indices.setdefault(get_patient_id(file), []).append(index)

    patient_ids = sorted(patient_to_indices)
    random.Random(seed).shuffle(patient_ids)

    val_count = max(1, int(len(patient_ids) * val_split))
    val_patients = set(patient_ids[:val_count])
    train_patients = set(patient_ids[val_count:])

    train_indices = [
        index
        for patient_id in train_patients
        for index in patient_to_indices[patient_id]
    ]
    val_indices = [
        index
        for patient_id in val_patients
        for index in patient_to_indices[patient_id]
    ]

    if not train_indices or not val_indices:
        raise ValueError("Need at least two patient groups to create train/validation splits")

    return DatasetSplit(
        train_indices=sorted(train_indices),
        val_indices=sorted(val_indices),
        train_patients=train_patients,
        val_patients=val_patients,
    )

def get_patient_id(file: str) -> str:
    return Path(file).stem.split("_", 1)[0]

def build_dataloaders(dataset: Dataset, split: DatasetSplit, train_config: dict,) -> tuple[DataLoader, DataLoader]:
    nw = train_config["NUM_WORKERS"]
    common = dict(
        batch_size=train_config["BATCH_SIZE"],
        num_workers=nw,
        pin_memory=True,
        persistent_workers=nw > 0,
        prefetch_factor=4 if nw > 0 else None,
    )
    train_loader = DataLoader(
        Subset(dataset, split.train_indices),
        shuffle=True, drop_last=False, **common,
    )
    val_loader = DataLoader(
        Subset(dataset, split.val_indices),
        shuffle=False, drop_last=False, **common,
    )
    return train_loader, val_loader

class RPPGDataset(Dataset):
    def __init__(self, files: list[str]):
        self.files = files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = np.load(self.files[index])
        patches = torch.from_numpy(sample["patches"]).float().permute(0, 1, 4, 2, 3).contiguous()
        ppg = torch.from_numpy(sample["ppg"]).float()
        return patches, ppg
