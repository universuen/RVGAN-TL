from math import sqrt, log, ceil

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from adapt.instance_based import TrAdaBoost as Base

import src.config.tr_ada_boost
from src import config
from src.config.tr_ada_boost import classifiers as n_classifier
from src.datasets import BasicDataset
from src.classifier import Classifier


class TrAdaBoost:
    def __init__(self):
        self.metrics = {
            'F1': .0,
            'G-Mean': .0,
            'AUC': .0,
        }
        self.model = Base(
            estimator=DecisionTreeClassifier(max_depth=5),
            n_estimators=src.config.tr_ada_boost.classifiers,
            random_state=src.config.seed,
        )

    def fit(self, src_dataset: BasicDataset, tgt_dataset: BasicDataset):
        src_dataset.to('cpu')
        tgt_dataset.to('cpu')
        src_x, src_y = src_dataset.samples.numpy(), src_dataset.labels.numpy()
        tgt_x, tgt_y = tgt_dataset.samples.numpy(), tgt_dataset.labels.numpy()
        self.model.fit(src_x, src_y, tgt_x, tgt_y)

    def predict(self, x: torch.Tensor):
        x = x.cpu().numpy()
        prediction = self.model.predict(x)
        return torch.from_numpy(prediction).to(src.config.device)

    def test(self, test_dataset: BasicDataset):
        with torch.no_grad():
            x, label = test_dataset.samples.cpu(), test_dataset.labels.cpu()
            predicted_label = self.predict(x).cpu()
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

# class TrAdaBoost:
#     def __init__(self):
#         self.classifiers = [Classifier(f'boosted_{i}') for i in range(n_classifier)]
#         self.betas: torch.Tensor = None
#         self.metrics = {
#             'F1': .0,
#             'G-Mean': .0,
#             'AUC': .0,
#         }
#
#     def fit(self, src_dataset: BasicDataset, tgt_dataset: BasicDataset):
#         final_betas = []
#         combined_dataset = BasicDataset()
#         src_dataset.to(config.device)
#         tgt_dataset.to(config.device)
#         combined_dataset.samples = torch.cat([src_dataset.samples, tgt_dataset.samples])
#         combined_dataset.labels = torch.cat([src_dataset.labels, tgt_dataset.labels])
#         weights = torch.ones(len(combined_dataset), device=config.device)
#         n = len(src_dataset)
#         m = len(tgt_dataset)
#         beta = 1 / (1 + sqrt(2 * log(n / n_classifier)))
#         for i in range(n_classifier):
#             p = weights / sum(weights)
#             classifier = self.classifiers[i]
#             classifier.fit(combined_dataset, p)
#             error_tgt = sum(
#                 weights[n:n + m] * abs(classifier.predict(tgt_dataset[:][0]) - tgt_dataset[:][1])
#             ) / sum(weights[n:n + m])
#             if error_tgt >= 0.5:
#                 error_tgt = 0.5 - 1e-3
#             beta_t = error_tgt / (1 - error_tgt)
#             betas = torch.tensor([beta if j < n else beta_t for j in range(n + m)], device=config.device)
#             signs = torch.tensor([1 if j < n else -1 for j in range(n + m)], device=config.device)
#             exponents = torch.cat(
#                 [
#                     abs(classifier.predict(src_dataset[:][0]) - src_dataset[:][1]),
#                     abs(classifier.predict(tgt_dataset[:][0]) - tgt_dataset[:][1]),
#                 ]
#             )
#             weights = weights * (betas ** (signs * exponents))
#             final_betas.append(beta_t)
#
#         # remove extra classifiers and coefficients
#         self.classifiers = self.classifiers[ceil(n_classifier / 2):n_classifier]
#         self.betas = torch.tensor(final_betas[ceil(n_classifier / 2):n_classifier], device=config.device)
#
#     def predict(self, x):
#         prediction = torch.stack([i.predict(x) for i in self.classifiers]).T
#         result = [
#             1 if i >= 0 else 0 for i in
#             torch.prod(self.betas ** -prediction, dim=1) - torch.prod(self.betas ** -0.5)
#         ]
#         return torch.tensor(result, device=config.device)
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
