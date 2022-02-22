import context

import random

import torch
import matplotlib.pyplot as plt

import src

FILE_NAME = 'wisconsin.dat'

if __name__ == '__main__':
    src.utils.prepare_dataset(FILE_NAME)
    r_dataset = src.datasets.RoulettePositiveDataset()
    r_dataset.samples = torch.tensor(list(range(len(r_dataset))))
    sample_cnt = dict()
    for i in range(len(r_dataset)):
        sample_cnt[r_dataset.samples[i].item()] = 0
    assert len(sample_cnt) == len(r_dataset)

    for i in r_dataset.get_roulette_samples(10000):
        sample_cnt[i.item()] += 1

    x = list(range(len(r_dataset)))
    _, (ax1, ax2) = plt.subplots(2)
    ax1.plot(x, r_dataset.fits, 'tab:blue')
    ax2.plot(x, sample_cnt.values(), 'tab:orange')
    ax1.set_xlabel('sample ID')
    ax2.set_xlabel('sample ID')
    ax1.set_ylabel('sample fit', color='tab:blue')
    ax2.set_ylabel('chosen frequency', color='tab:orange')
    plt.show()
