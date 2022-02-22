from math import sqrt

import torch
import numpy as np
from torch.nn.functional import binary_cross_entropy
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

from src import config, logger, models
from src.datasets import BasicDataset


class Classifier:
    def __init__(self, name: str):
        self.name = name
        self.model = DecisionTreeClassifier(max_depth=5)
        self.logger = logger.Logger(name)
        self.metrics = {
            'F1': .0,
            'G-Mean': .0,
            'AUC': .0,
        }

    def fit(self, dataset: BasicDataset, weights: torch.Tensor = None):
        self.logger.info('Started training')

        x, y = dataset.samples.cpu().numpy(), dataset.labels.cpu().numpy()
        if weights is not None:
            weights = weights.numpy()
        self.model.fit(x, y, weights)

        self.logger.info('Finished training')

    def predict(self, x):
        return self.model.predict(x.cpu().numpy())

    def test(self, test_dataset: BasicDataset):
        with torch.no_grad():
            x, label = test_dataset.samples.cpu(), test_dataset.labels.cpu()
            predicted_label = self.predict(x)
            tn, fp, fn, tp = confusion_matrix(
                y_true=label,
                y_pred=predicted_label,
            ).ravel()

            precision = tp / (tp + fp) if tp + fp != 0 else 0
            recall = tp / (tp + fn) if tp + fn != 0 else 0
            specificity = tn / (tn + fp) if tn + fp != 0 else 0

            f1 = 2 * recall * precision / (recall + precision) if recall + precision != 0 else 0
            g_mean = sqrt(recall * specificity)

            auc = roc_auc_score(
                y_true=label,
                y_score=predicted_label,
            )

            self.metrics['F1'] = f1
            self.metrics['G-Mean'] = g_mean
            self.metrics['AUC'] = auc

    @staticmethod
    def _prob2label(prob):
        probabilities = prob.squeeze(dim=1)
        labels = np.zeros(probabilities.size())
        for i, p in enumerate(probabilities):
            if p >= 0.5:
                labels[i] = 1
        return torch.from_numpy(labels).to(config.device)


# class Classifier:
#     def __init__(self, name: str):
#         self.name = name
#         self.model = models.ClassifierModel().to(config.device)
#         self.logger = logger.Logger(name)
#         self.metrics = {
#             'F1': .0,
#             'G-Mean': .0,
#             'AUC': .0,
#         }
#
#     def fit(self, dataset: BasicDataset, weights: torch.Tensor = None):
#         self.logger.info('Started training')
#         self.logger.debug(f'Using device: {config.device}')
#
#         weights = torch.ones(len(dataset), device=config.device) if weights is None else weights.to(config.device)
#         optimizer = Adam(
#             params=self.model.parameters(),
#             lr=config.classifier.lr,
#             betas=(0.5, 0.9),
#         )
#
#         x, label = dataset.samples.to(config.device), dataset.labels.to(config.device)
#         for _ in range(config.classifier.epochs):
#             self.model.zero_grad()
#             prediction = self.model(x).squeeze()
#             loss = binary_cross_entropy(
#                 input=prediction,
#                 target=label,
#                 weight=weights,
#             )
#             loss.backward()
#             optimizer.step()
#
#         self.model.eval()
#         self.logger.info('Finished training')
#
#     def predict(self, x):
#         x = x.to(config.device)
#         prob = self.model(x)
#         return self._prob2label(prob)
#
#     def test(self, test_dataset: BasicDataset):
#         with torch.no_grad():
#             x, label = test_dataset.samples.cpu(), test_dataset.labels.cpu()
#             predicted_label = self.predict(x).cpu()
#             tn, fp, fn, tp = confusion_matrix(
#                 y_true=label,
#                 y_pred=predicted_label,
#             ).ravel()
#
#             precision = tp / (tp + fp) if tp + fp != 0 else 0
#             recall = tp / (tp + fn) if tp + fn != 0 else 0
#             specificity = tn / (tn + fp) if tn + fp != 0 else 0
#
#             f1 = 2 * recall * precision / (recall + precision) if recall + precision != 0 else 0
#             g_mean = sqrt(recall * specificity)
#
#             auc = roc_auc_score(
#                 y_true=label,
#                 y_score=predicted_label,
#             )
#
#             self.metrics['F1'] = f1
#             self.metrics['G-Mean'] = g_mean
#             self.metrics['AUC'] = auc
#
#     @staticmethod
#     def _prob2label(prob):
#         probabilities = prob.squeeze(dim=1)
#         labels = np.zeros(probabilities.size())
#         for i, p in enumerate(probabilities):
#             if p >= 0.5:
#                 labels[i] = 1
#         return torch.from_numpy(labels).to(config.device)
