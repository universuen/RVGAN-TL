import torch

import src
from src import config, models
from src.models import SNGANGModel, SNGANDModel
from src.datasets import PositiveDataset, RoulettePositiveDataset
from ._base import Base


class RVSNGAN(Base):
    def __init__(self):
        super().__init__(SNGANGModel(), SNGANDModel())

    def _fit(self):
        d_optimizer = torch.optim.Adam(
            params=self.d.parameters(),
            lr=config.gan.d_lr,
            betas=(0.5, 0.999),
        )
        g_optimizer = torch.optim.Adam(
            params=self.g.parameters(),
            lr=config.gan.g_lr,
            betas=(0.5, 0.999),
        )

        x = RoulettePositiveDataset().get_roulette_samples(len(PositiveDataset())).to(config.device)
        for _ in range(config.gan.epochs):
            for __ in range(config.gan.d_loops):
                self.d.zero_grad()
                prediction_real = self.d(x)
                loss_real = - prediction_real.mean()
                z = torch.randn(len(x), models.z_size, device=config.device)
                fake_x = self.g(z).detach()
                prediction_fake = self.d(fake_x)
                loss_fake = prediction_fake.mean()
                loss = loss_real + loss_fake
                loss.backward()
                d_optimizer.step()
            for __ in range(config.gan.g_loops):
                self.g.zero_grad()
                real_x_hidden_output = self.d.hidden_output.detach()
                z = torch.randn(len(x), models.z_size, device=config.device)
                fake_x = self.g(z)
                final_output = self.d(fake_x)
                fake_x_hidden_output = self.d.hidden_output
                real_x_hidden_distribution = src.utils.normalize(real_x_hidden_output)
                fake_x_hidden_distribution = src.utils.normalize(fake_x_hidden_output)
                hidden_loss = torch.norm(
                    real_x_hidden_distribution - fake_x_hidden_distribution,
                    p=2
                ) * config.gan.hl_lambda
                loss = -final_output.mean() + hidden_loss
                loss.backward()
                g_optimizer.step()
