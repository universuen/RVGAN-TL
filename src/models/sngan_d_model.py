import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

import src


class SNGANDModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.step_1 = nn.Sequential(
            spectral_norm(nn.Linear(src.models.x_size, 512)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(512, 128)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(128, 32)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(32, 8)),
            nn.LeakyReLU(0.2),
        )
        self.step_2 = spectral_norm(nn.Linear(8, 1))
        self.hidden_output = None

        self.apply(src.utils.init_weights)

    def forward(self, x: torch.Tensor):
        self.hidden_output = self.step_1(x)
        output = self.step_2(self.hidden_output)
        return output
