import context

import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare
import matplotlib.pyplot as plt

import src
from scripts.datasets import DATASETS

METHODS = [
    'Original',
    'RandomOverSampler',
    'SMOTE',
    'ADASYN',
    'BorderlineSMOTE',
    'RVGAN-TL',
]

METRICS = [
    'F1',
    'AUC',
    'G-Mean',
]


def cal_cd(n, k, q):
    return q * (np.sqrt(k * (k + 1) / (6 * n)))


excel_path = src.config.path.test_results / 'vstm_final.xlsx'

df = pd.read_excel(excel_path, index_col=0, sheet_name=None)

ranks = {k: {kk: [] for kk in METHODS} for k in METRICS}

for i in METRICS:
    sheet = df[i]
    scores = []
    rank = sheet.T.rank(ascending=False)
    for j in METHODS:
        scores.append(list(sheet[j]))
        ranks[i][j] = sum(rank.loc[j]) / len(rank.loc[j])
    print(friedmanchisquare(*scores))

cd = cal_cd(len(DATASETS), 6, 2.85)

for i in METRICS:
    plt.subplots_adjust(left=0.25)
    x = list(ranks[i].values())
    y = [j if j != 'RandomOverSampler' else 'ROS' for j in ranks[i].keys()]
    min_ = [i for i in x - cd / 2]
    max_ = [i for i in x + cd / 2]
    plt.title(f'{i} Friedman Test Result')
    plt.scatter(x, y)
    plt.hlines(y, min_, max_)
    plt.show()
