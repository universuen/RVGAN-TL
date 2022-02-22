import context

import random

import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import src

TARGET_GAN = src.gans.RGAN

if __name__ == '__main__':
    result = dict()
    src.utils.set_random_state()
    datasets = random.choices(
        population=[
            'dermatology-6.dat',
            'ecoli-0-1-3-7_vs_2-6.dat',
            'ecoli-0-1-4-6_vs_5.dat',
            'ecoli-0-1-4-7_vs_2-3-5-6.dat',
            'ecoli-0-1-4-7_vs_5-6.dat',
            'ecoli-0-1_vs_2-3-5.dat',
            'ecoli-0-1_vs_5.dat',
            'ecoli-0-2-3-4_vs_5.dat',
            'ecoli-0-2-6-7_vs_3-5.dat',
            'ecoli-0-3-4-6_vs_5.dat',
            'ecoli-0-3-4-7_vs_5-6.dat',
            'ecoli-0-3-4_vs_5.dat',
            'ecoli-0-4-6_vs_5.dat',
            'ecoli-0-6-7_vs_3-5.dat',
            'ecoli-0-6-7_vs_5.dat',
            'ecoli-0_vs_1.dat',
            'ecoli1.dat',
            'ecoli2.dat',
            'ecoli3.dat',
            'ecoli4.dat',
            'glass-0-1-2-3_vs_4-5-6.dat',
            'glass-0-1-4-6_vs_2.dat',
            'glass-0-1-5_vs_2.dat',
            'glass-0-1-6_vs_2.dat',
            'glass-0-1-6_vs_5.dat',
            'glass-0-4_vs_5.dat',
            'glass-0-6_vs_5.dat',
            'glass0.dat',
            'glass1.dat',
            'glass2.dat',
            'glass4.dat',
            'glass5.dat',
            'glass6.dat',
            'haberman.dat',
            'iris0.dat',
            'led7digit-0-2-4-5-6-7-8-9_vs_1.dat',
            'new-thyroid1.dat',
            'newthyroid2.dat',
            'page-blocks-1-3_vs_4.dat',
            'page-blocks0.dat',
            'pima.dat',
            'poker-8-9_vs_5.dat',
            'poker-8-9_vs_6.dat',
            'poker-8_vs_6.dat',
            'poker-9_vs_7.dat',
            'segment0.dat',
            'shuttle-2_vs_5.dat',
            'shuttle-6_vs_2-3.dat',
            'shuttle-c0-vs-c4.dat',
            'shuttle-c2-vs-c4.dat',
            'vehicle0.dat',
            'vehicle1.dat',
            'vehicle2.dat',
            'vehicle3.dat',
            'vowel0.dat',
            'winequality-red-3_vs_5.dat',
            'winequality-red-4.dat',
            'winequality-red-8_vs_6-7.dat',
            'winequality-red-8_vs_6.dat',
            'winequality-white-3-9_vs_5.dat',
            'winequality-white-3_vs_7.dat',
            'winequality-white-9_vs_4.dat',
            'wisconsin.dat',
            'yeast-0-2-5-6_vs_3-7-8-9.dat',
            'yeast-0-2-5-7-9_vs_3-6-8.dat',
            'yeast-0-3-5-9_vs_7-8.dat',
            'yeast-0-5-6-7-9_vs_4.dat',
            'yeast-1-2-8-9_vs_7.dat',
            'yeast-1-4-5-8_vs_7.dat',
            'yeast-1_vs_7.dat',
            'yeast-2_vs_4.dat',
            'yeast-2_vs_8.dat',
            'yeast1.dat',
            'yeast3.dat',
            'yeast4.dat',
            'yeast5.dat',
            'yeast6.dat',
        ],
        k=9,
    )

    for dataset_name in datasets:
        print(f'***********************{dataset_name}***********************')
        src.utils.prepare_dataset(dataset_name)
        dataset = src.datasets.FullDataset(training=True)
        raw_x, raw_y = dataset[:]
        raw_x = raw_x.numpy()
        raw_y = raw_y.numpy()

        src.utils.set_random_state()
        gan = TARGET_GAN()
        gan.fit()
        z = torch.randn([len(raw_y) - int(2 * sum(raw_y)), src.models.z_size], device=src.config.device)
        x = np.concatenate(
            [raw_x, gan.g(z).detach().cpu().numpy()],
        )
        y = np.concatenate([raw_y, np.full(len(x) - len(raw_x), 2)])
        embedded_x = TSNE(
            learning_rate='auto',
            init='random',
            random_state=src.config.seed,
        ).fit_transform(x)
        result[dataset_name] = [embedded_x, y]

    sns.set_theme()
    plt.close('all')
    _, axes = plt.subplots(3, 3)
    for (key, value), axe in zip(result.items(), axes.flat):
        axe.set(xticklabels=[])
        axe.set(yticklabels=[])
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
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
    plt.show()
