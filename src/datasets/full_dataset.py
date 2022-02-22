from typing import Callable

import torch

from src import datasets


class FullDataset(datasets.BasicDataset):
    def __init__(self, training: bool = True, transform: Callable = None, target_transform: Callable = None):
        super().__init__(training, transform, target_transform)
        self.samples = datasets.training_samples if training else datasets.test_samples
        self.samples = torch.from_numpy(self.samples).float()
        self.labels = datasets.training_labels if training else datasets.test_labels
        self.labels = torch.from_numpy(self.labels).float()
