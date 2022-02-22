from numpy import ndarray

from .basic_dataset import BasicDataset
from .full_dataset import FullDataset
from .negative_dataset import NegativeDataset
from .positive_dataset import PositiveDataset
from .roulette_positive_dataset import RoulettePositiveDataset


training_samples: ndarray = None
training_labels: ndarray = None
test_samples: ndarray = None
test_labels: ndarray = None
