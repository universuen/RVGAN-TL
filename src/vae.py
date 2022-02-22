import random

import torch
from torch.nn.functional import mse_loss

from src import config
from src.datasets import PositiveDataset
from src.logger import Logger
from src.models import VAEEModel, VAEDModel


class VAE:

    def __init__(self):
        self.logger = Logger(self.__class__.__name__)
        self.e = VAEEModel().to(config.device)
        self.d = VAEDModel().to(config.device)

    def fit(self):
        self.logger.info('Started training')
        self.logger.debug(f'Using device: {config.device}')
        e_optimizer = torch.optim.Adam(
            params=self.e.parameters(),
            lr=config.vae.e_lr,
        )
        d_optimizer = torch.optim.Adam(
            params=self.d.parameters(),
            lr=config.vae.d_lr,
        )

        x = PositiveDataset()[:][0].to(config.device)
        for _ in range(config.vae.epochs):
            # clear gradients
            self.e.zero_grad()
            self.d.zero_grad()
            # calculate z, mu and sigma
            z, mu, sigma = self.e(x)
            # calculate x_hat
            x_hat = self.d(z)
            # calculate loss
            divergence = - 0.5 * torch.sum(1 + torch.log(sigma ** 2) - mu ** 2 - sigma ** 2)
            loss = divergence + mse_loss(x_hat, x)
            # calculate gradients
            loss.backward()
            # optimize models
            e_optimizer.step()
            d_optimizer.step()

        self.e.eval()
        self.d.eval()
        self.logger.info("Finished training")

    def generate_z(self, size: int = 1):
        seeds = torch.stack(random.choices(PositiveDataset().samples, k=size)).to(config.device)
        return self.e(seeds)[0]
