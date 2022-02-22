import context

from src import utils
from src.datasets import FullDataset
from src.classifier import Classifier

FILE_NAME = 'pima.dat'

if __name__ == '__main__':
    # prepare dataset
    utils.prepare_dataset(FILE_NAME)

    # normally train
    utils.set_random_state()
    classifier = Classifier('Test_Normal_Train')
    classifier.fit(
        dataset=FullDataset(training=True),
    )
    classifier.test(FullDataset(training=False))
    for name, value in classifier.metrics.items():
        print(f'{name:<15}:{value:>10.4f}')
