from typing import Callable

import torch

from src.datasets.full_dataset import FullDataset


class PositiveDataset(FullDataset):
    def __init__(
            self,
            training: bool = True,
            transform: Callable = None,
            target_transform: Callable = None,
    ):
        super().__init__(training, transform, target_transform)
        target_item_indices = []

        for idx, label in enumerate(self.labels):
            if label == 1:
                target_item_indices.append(idx)
        self.samples = self.samples.numpy()
        self.labels = self.labels.numpy()
        self.samples = self.samples[target_item_indices]
        self.labels = self.labels[target_item_indices]
        self.samples = torch.from_numpy(self.samples)
        self.labels = torch.from_numpy(self.labels)
