from abc import abstractmethod

import torch
from torch.nn import Module

from src import config
from src.logger import Logger


class Base:
    def __init__(
            self,
            g: Module,
            d: Module,
    ):
        self.logger = Logger(self.__class__.__name__)
        self.g = g.to(config.device)
        self.d = d.to(config.device)

    def fit(self):
        self.logger.info('Started training')
        self.logger.debug(f'Using device: {config.device}')
        self._fit()
        self.g.eval()
        self.d.eval()
        self.logger.info(f'Finished training')

    @abstractmethod
    def _fit(self):
        pass

    def generate_samples(self, z: torch.Tensor):
        return self.g(z)
