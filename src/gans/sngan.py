import torch

from src import config, models
from src.models import SNGANGModel, SNGANDModel
from src.datasets import PositiveDataset
from ._base import Base


class SNGAN(Base):
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

        x = PositiveDataset()[:][0].to(config.device)
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
                z = torch.randn(len(x), models.z_size, device=config.device)
                fake_x = self.g(z)
                prediction = self.d(fake_x)
                loss = - prediction.mean()
                loss.backward()
                g_optimizer.step()
