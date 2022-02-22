from typing import Callable

import torch
from torch.utils.data import Dataset

import src.config


class BasicDataset(Dataset):
    def __init__(
            self,
            training: bool = True,
            transform: Callable = None,
            target_transform: Callable = None,
    ):
        self.samples: torch.Tensor = None
        self.labels: torch.Tensor = None
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        sample = self.samples[item]
        label = self.labels[item]
        sample = self.transform(sample) if self.transform else sample
        label = self.target_transform(label) if self.target_transform else label
        return sample, label

    def to(self, device: str):
        self.samples = self.samples.to(device)
        self.labels = self.labels.to(device)
        return self
