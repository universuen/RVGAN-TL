import os

import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE

import src
from scripts.datasets import DATASETS

TEST_NAME = '2-14'

TRADITIONAL_METHODS = [
    RandomOverSampler,
    SMOTE,
    ADASYN,
    BorderlineSMOTE,
]

RGAN = src.gans.RVSNGAN

K = 5

METRICS = [
    'F1',
    'AUC',
    'G-Mean',
]


if __name__ == '__main__':
    src.config.logger.level = 'WARNING'
    result_file = src.config.path.test_results / f'vstm_{TEST_NAME}.xlsx'
    if os.path.exists(result_file):
        input(f'{result_file} already existed, continue?')
    all_methods = ['Original', *[i.__name__ for i in TRADITIONAL_METHODS], 'RGAN-TL']
    result = {
        k: pd.DataFrame(
            {
                kk:
                    {
                        kkk: 0.0 for kkk in [*DATASETS, 'mean']
                    } for kk in all_methods
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
                kk: [] for kk in all_methods
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
            # test original classifier
            src.utils.set_random_state()
            o_classifier = src.classifier.Classifier('Original')
            o_classifier.fit(training_dataset)
            o_classifier.test(test_dataset)
            for metric_name in METRICS:
                temp_result[metric_name]['Original'].append(o_classifier.metrics[metric_name])
            # test traditional methods
            for METHOD in TRADITIONAL_METHODS:
                try:
                    x, y = training_dataset.samples, training_dataset.labels
                    x = x.numpy()
                    y = y.numpy()
                    x, y = METHOD(random_state=src.config.seed).fit_resample(x, y)
                    balanced_dataset = src.datasets.BasicDataset()
                    balanced_dataset.samples = torch.from_numpy(x)
                    balanced_dataset.labels = torch.from_numpy(y)
                    src.utils.set_random_state()
                    tm_classifier = src.classifier.Classifier(METHOD.__name__)
                    tm_classifier.fit(balanced_dataset)
                    tm_classifier.test(test_dataset)
                    for metric_name in METRICS:
                        temp_result[metric_name][METHOD.__name__].append(tm_classifier.metrics[metric_name])
                except (RuntimeError, ValueError):
                    for metric_name in METRICS:
                        temp_result[metric_name][METHOD.__name__].append(0)
            # test RGAN-TL
            src.utils.set_random_state()
            rgan_dataset = src.utils.get_rgan_dataset(RGAN())
            esb_classifier = src.tr_ada_boost.TrAdaBoost()
            esb_classifier.fit(rgan_dataset, training_dataset)
            esb_classifier.test(test_dataset)
            for metric_name in METRICS:
                temp_result[metric_name]['RGAN-TL'].append(esb_classifier.metrics[metric_name])
        # calculate final metrics
        for method_name in all_methods:
            for metric_name in METRICS:
                result[metric_name][method_name][dataset_name] = np.mean(temp_result[metric_name][method_name])
        # calculate average metrics on all datasets
        for gan_name in all_methods:
            for metric_name in METRICS:
                result[metric_name][gan_name]['mean'] = np.mean([i for i in result[metric_name][gan_name].values])
        # write down current result
        with pd.ExcelWriter(result_file) as writer:
            for metric_name in METRICS:
                df = result[metric_name]
                df.to_excel(writer, metric_name)
                df.style.highlight_max(axis=1).to_excel(writer, metric_name, float_format='%.4f')
