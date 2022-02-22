import random

import torch
import numpy as np
import pandas as pd
from torch import nn
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split

import src


def init_weights(layer: nn.Module):
    layer_name = layer.__class__.__name__
    if 'Linear' in layer_name:
        nn.init.normal_(layer.weight.data, 0.0, 0.02)
        if layer.bias is not None:
            nn.init.constant_(layer.bias.data, 0)
    elif layer_name == 'BatchNorm1d':
        nn.init.normal_(layer.weight.data, 1.0, 0.02)
        nn.init.constant_(layer.bias.data, 0)


def set_random_state(seed: int = None) -> None:
    if seed is None:
        seed = src.config.seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def preprocess_data(file_name: str) -> (np.ndarray, np.ndarray):
    set_random_state()
    # concatenate the file path
    file_path = src.config.path.datasets / file_name
    # calculate skip rows
    skip_rows = 0
    with open(file_path, 'r') as f:
        while True:
            line = f.readline()
            if line[0] != '@':
                break
            else:
                skip_rows += 1
    # read raw data
    df = pd.read_csv(file_path, sep=',', skiprows=skip_rows, header=None)
    np_array = df.to_numpy()
    np.random.shuffle(np_array)
    # partition labels and samples
    labels = np_array[:, -1].copy()
    samples = np_array[:, :-1].copy()
    # digitize labels
    for i, _ in enumerate(labels):
        labels[i] = labels[i].strip()
    labels[labels[:] == 'positive'] = 1
    labels[labels[:] == 'negative'] = 0
    labels = labels.astype('int')
    # normalize samples
    samples = minmax_scale(samples.astype('float32'))
    src.models.x_size = samples.shape[1]
    return samples, labels


def prepare_dataset(name: str, training_test_ratio: float = 0.8) -> None:
    samples, labels = preprocess_data(name)
    training_samples, test_samples, training_labels, test_labels = train_test_split(
        samples,
        labels,
        train_size=training_test_ratio,
        random_state=src.config.seed,
    )
    src.datasets.training_samples = training_samples
    src.datasets.training_labels = training_labels
    src.datasets.test_samples = test_samples
    src.datasets.test_labels = test_labels


def get_final_test_metrics(statistics: dict) -> dict:
    metrics = dict()
    for name, values in statistics.items():
        if name == 'Loss':
            continue
        else:
            metrics[name] = values[-1]
    return metrics


def normalize(x: torch.Tensor) -> torch.Tensor:
    return (x - x.min()) / (x.max() - x.min())


def get_knn_indices(sample: torch.Tensor, all_samples: torch.Tensor, k: int = 5) -> torch.Tensor:
    dist = torch.empty(len(all_samples))
    for i, v in enumerate(all_samples):
        dist[i] = torch.norm(sample - v, p=2)
    return torch.topk(dist, k, largest=False).indices


def get_gan_dataset(gan: src.gans.GANLike) -> src.datasets.FullDataset:

    gan.fit()

    full_dataset = src.datasets.FullDataset().to(src.config.device)
    pos_dataset = src.datasets.PositiveDataset().to(src.config.device)
    neg_dataset = src.datasets.NegativeDataset().to(src.config.device)
    target_dataset = src.datasets.FullDataset().to(src.config.device)
    # generate positive samples until reaching balance
    total_pos_cnt = len(pos_dataset)
    total_neg_cnt = len(neg_dataset)

    target_sample_num = total_neg_cnt - total_pos_cnt
    if target_sample_num <= 0:
        return full_dataset
    z = torch.rand(target_sample_num, src.models.z_size, device=src.config.device)
    new_samples = gan.generate_samples(z)
    new_labels = torch.ones(target_sample_num, device=src.config.device)
    target_dataset.samples = torch.cat(
        [
            target_dataset.samples,
            new_samples,
        ],
    )
    target_dataset.labels = torch.cat(
        [
            target_dataset.labels,
            new_labels,
        ]
    )
    target_dataset.samples = target_dataset.samples.detach()
    target_dataset.labels = target_dataset.labels.detach()
    return target_dataset


def get_rgan_dataset(rgan: src.gans.GANLike) -> src.datasets.FullDataset:
    vae = src.vae.VAE()
    vae.fit()
    rgan.fit()

    full_dataset = src.datasets.FullDataset().to(src.config.device)
    pos_dataset = src.datasets.PositiveDataset().to(src.config.device)
    neg_dataset = src.datasets.NegativeDataset().to(src.config.device)

    # count negative samples in the overlapping area
    ol_neg_cnt = 0
    for i in neg_dataset.samples:
        indices = get_knn_indices(i, full_dataset.samples)
        labels = full_dataset.labels[indices]
        if 1 in labels:
            ol_neg_cnt += 1

    # count positive samples in the overlapping area
    ol_pos_cnt = 0
    for i in pos_dataset.samples:
        indices = get_knn_indices(i, full_dataset.samples)
        labels = full_dataset.labels[indices]
        if 0 in labels:
            ol_pos_cnt += 1

    target_dataset = src.datasets.FullDataset().to(src.config.device)
    # generate positive samples until reaching balance
    total_pos_cnt = len(pos_dataset)
    total_neg_cnt = len(neg_dataset)
    while True:
        if total_pos_cnt >= total_neg_cnt or ol_pos_cnt >= ol_neg_cnt:
            break
        else:
            # update the number of positive samples
            z = vae.generate_z()
            new_sample = rgan.generate_samples(z)
            new_label = torch.tensor([1], device=src.config.device)
            target_dataset.samples = torch.cat(
                [
                    target_dataset.samples,
                    new_sample,
                ],
            )
            target_dataset.labels = torch.cat(
                [
                    target_dataset.labels,
                    new_label,
                ]
            )
            total_pos_cnt += 1
            # update the number of overlapping positive samples
            indices = get_knn_indices(new_sample, full_dataset.samples)
            labels = full_dataset.labels[indices]
            if 0 in labels:
                ol_pos_cnt += 1
    target_dataset.samples = target_dataset.samples.detach()
    target_dataset.labels = target_dataset.labels.detach()
    return target_dataset
