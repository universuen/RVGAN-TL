import torch
from torch import nn

import src


class VAEEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.calculate_mu = nn.Sequential(
            nn.Linear(src.models.x_size, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, src.models.z_size),
        )
        self.calculate_log_variance = nn.Sequential(
            nn.Linear(src.models.x_size, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, src.models.z_size),
        )
        self.apply(src.utils.init_weights)

    def forward(self, x: torch.Tensor):
        mu = self.calculate_mu(x)
        log_variance = self.calculate_log_variance(x)
        sigma = torch.exp(0.5 * log_variance)
        epsilon = torch.randn_like(mu)
        z = epsilon * sigma + mu
        return z, mu, sigma
