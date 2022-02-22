import context

from src import utils, config
from src.datasets import PositiveDataset
from src.vae import VAE


FILE_NAME = 'segment0.dat'

if __name__ == '__main__':
    # prepare dataset
    utils.prepare_dataset(FILE_NAME)
    # train
    dataset = PositiveDataset()
    utils.set_random_state()
    vae = VAE()
    vae.train(dataset=dataset)
    # test
    x = dataset[:3][0].to(config.device)
    z, mu, sigma = vae.e(x)
    print(mu.mean().mean().item())
    print(sigma.mean().mean().item())
