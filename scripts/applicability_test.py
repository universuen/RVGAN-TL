import os

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

import src
from scripts.datasets import DATASETS

TEST_NAME = '2-11'

PAIRS = [
    (src.gans.GAN, src.gans.RVGAN),
    (src.gans.WGAN, src.gans.RVWGAN),
    (src.gans.WGANGP, src.gans.RVWGANGP),
    (src.gans.SNGAN, src.gans.RVSNGAN),
]

K = 5

METRICS = [
    'F1',
    'AUC',
    'G-Mean',
]


def highlight_higher_cells(s: pd.Series) -> list[str]:
    result_ = []
    for i_1, i_2 in zip(s[0::2], s[1::2]):
        if i_1 > i_2:
            result_.append('background-color: yellow')
            result_.append('')
        elif i_1 < i_2:
            result_.append('')
            result_.append('background-color: yellow')
        else:
            result_.append('')
            result_.append('')
    return result_


if __name__ == '__main__':
    src.config.logger.level = 'WARNING'
    result_file = src.config.path.test_results / f'applicability_{TEST_NAME}.xlsx'
    if os.path.exists(result_file):
        input(f'{result_file} already existed, continue?')
    all_gans = []
    for i, j in PAIRS:
        all_gans.append(i.__name__)
        all_gans.append(j.__name__)
    result = {
        k: pd.DataFrame(
            {
                kk:
                    {
                        kkk: 0.0 for kkk in [*DATASETS, 'mean']
                    } for kk in all_gans
            }
        ) for k in METRICS
    }

    for dataset_name in tqdm(DATASETS):
        # prepare data
        src.utils.set_random_state()
        samples, labels = src.utils.preprocess_data(dataset_name)
        skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=src.config.seed)
        temp_result = {
            k: {
                kk: [] for kk in all_gans
            } for k in METRICS
        }
        # k-fold test
        for training_indices, test_indices in skf.split(samples, labels):
            src.datasets.training_samples = samples[training_indices]
            src.datasets.training_labels = labels[training_indices]
            src.datasets.test_samples = samples[test_indices]
            src.datasets.test_labels = labels[test_indices]
            training_dataset = src.datasets.FullDataset(training=True)
            test_dataset = src.datasets.FullDataset(training=False)
            for GAN, RGAN in PAIRS:
                # test GAN
                src.utils.set_random_state()
                gan_dataset = src.utils.get_gan_dataset(GAN())
                gan_classifier = src.classifier.Classifier(GAN.__name__)
                gan_classifier.fit(gan_dataset)
                gan_classifier.test(test_dataset)
                for metric_name in METRICS:
                    temp_result[metric_name][GAN.__name__].append(gan_classifier.metrics[metric_name])
                # test RGAN
                src.utils.set_random_state()
                rgan_dataset = src.utils.get_rgan_dataset(RGAN())
                rgan_classifier = src.tr_ada_boost.TrAdaBoost()
                rgan_classifier.fit(rgan_dataset, training_dataset)
                rgan_classifier.test(test_dataset)
                for metric_name in METRICS:
                    temp_result[metric_name][RGAN.__name__].append(rgan_classifier.metrics[metric_name])
            # calculate final metrics
            for gan_name in all_gans:
                for metric_name in METRICS:
                    result[metric_name][gan_name][dataset_name] = np.mean(temp_result[metric_name][gan_name])
            # calculate average metrics on all datasets
            for gan_name in all_gans:
                for metric_name in METRICS:
                    result[metric_name][gan_name]['mean'] = np.mean([i for i in result[metric_name][gan_name].values])
            # write down current result
            with pd.ExcelWriter(result_file) as writer:
                for metric_name in METRICS:
                    df = result[metric_name]
                    df.to_excel(writer, metric_name)
                    df.style.apply(highlight_higher_cells, axis=1).to_excel(writer, metric_name, float_format='%.4f')
