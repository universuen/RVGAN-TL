from src.tr_ada_boost import TrAdaBoost
from src import datasets, utils


FILE_MANE = 'wisconsin.dat'


if __name__ == '__main__':
    utils.prepare_dataset('pima.dat')
    src_dataset = datasets.FullDataset()
    dst_dataset = datasets.FullDataset()
    esb = TrAdaBoost()
    esb.fit(src_dataset, dst_dataset)
    prediction = esb.predict(datasets.NegativeDataset()[:][0])
    print(prediction)
    esb.test(datasets.FullDataset(training=False))
    for name, value in esb.metrics.items():
        print(f'{name:<15}:{value:>10.4f}')
