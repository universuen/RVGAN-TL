import context

import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, RandomOverSampler

import src

DATASET = 'wisconsin.dat'

TRADITIONAL_METHODS = [
    RandomOverSampler,
    SMOTE,
    ADASYN,
    BorderlineSMOTE,
]

GAN_MODELS = [
    src.gans.GAN,
    src.gans.WGAN,
    src.gans.WGANGP,
    src.gans.SNGAN,
    src.gans.RVGAN,
    src.gans.RVWGAN,
    src.gans.RVWGANGP,
    src.gans.RVSNGAN,
]

if __name__ == '__main__':
    result = dict()
    src.utils.set_random_state()
    src.utils.prepare_dataset(DATASET)
    dataset = src.datasets.FullDataset(training=True)

    raw_x, raw_y = dataset[:]
    raw_x = raw_x.numpy()
    raw_y = raw_y.numpy()

    for M in TRADITIONAL_METHODS:
        x, _ = M(random_state=src.config.seed).fit_resample(raw_x, raw_y)
        y = np.concatenate([raw_y, np.full(len(x) - len(raw_x), 2)])
        embedded_x = TSNE(
            learning_rate='auto',
            init='random',
            random_state=src.config.seed,
        ).fit_transform(x)
        result[M.__name__] = [embedded_x, y]

    for M in GAN_MODELS:
        src.utils.set_random_state()
        gan = M()
        gan.fit()
        z = torch.randn([len(raw_y) - int(2 * sum(raw_y)), src.models.z_size], device=src.config.device)
        x = np.concatenate([raw_x, gan.g(z).detach().cpu().numpy()])
        y = np.concatenate([raw_y, np.full(len(x) - len(raw_x), 2)])
        embedded_x = TSNE(
            learning_rate='auto',
            init='random',
            random_state=src.config.seed,
        ).fit_transform(x)
        result[M.__name__] = [embedded_x, y]

    sns.set_style('white')
    fig, axes = plt.subplots(3, 4)
    for (key, value), axe in zip(result.items(), axes.flat):
        # axe.set(xticklabels=[])
        # axe.set(yticklabels=[])
        axe.set(title=key)
        majority = []
        minority = []
        generated_data = []
        for i, j in zip(value[0], value[1]):
            if j == 0:
                majority.append(i)
            elif j == 1:
                minority.append(i)
            else:
                generated_data.append(i)
        minority = np.array(minority)
        majority = np.array(majority)
        generated_data = np.array(generated_data)
        sns.scatterplot(
            x=majority[:, 0],
            y=majority[:, 1],
            ax=axe,
            alpha=0.5,
            label='majority',
        )
        sns.scatterplot(
            x=generated_data[:, 0],
            y=generated_data[:, 1],
            ax=axe,
            alpha=0.5,
            label='generated_data',
        )
        sns.scatterplot(
            x=minority[:, 0],
            y=minority[:, 1],
            ax=axe,
            alpha=1.0,
            s=10,
            label='minority',
        )
        axe.get_legend().remove()
    fig.set_size_inches(18, 10)
    fig.set_dpi(100)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.savefig(src.config.path.test_results / 'all_distribution.jpg')
    plt.show()
