import random
from typing import Callable

import numpy as np
import torch

from src.datasets import PositiveDataset, NegativeDataset


class RoulettePositiveDataset(PositiveDataset):
    def __init__(
            self,
            training: bool = True,
            transform: Callable = None,
            target_transform: Callable = None
    ):
        super().__init__(training, transform, target_transform)
        # calculate fits
        pos_samples = PositiveDataset(training, transform, target_transform)[:][0]
        neg_samples = NegativeDataset(training, transform, target_transform)[:][0]
        dist = np.zeros([len(pos_samples), len(neg_samples)])

        # calculate distances
        for i, minority_item in enumerate(pos_samples):
            for j, majority_item in enumerate(neg_samples):
                dist[i][j] = torch.norm(minority_item - majority_item, p=2) + 1e-3

        self.fits = 1 / dist.min(axis=1, initial=None)
        self.fits = self.fits / self.fits.sum()

    def get_roulette_samples(self, size: int) -> torch.Tensor:
        return torch.stack(
            random.choices(
                self.samples,
                weights=self.fits,
                k=size,
            )
        )
