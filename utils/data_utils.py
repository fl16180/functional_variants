import numpy as np
import pandas as pd
from sklearn.utils import resample


def split_train_test(data, test_frac=0.2, seed=None):
    if seed:
        np.random.seed(seed)

    m  = data.shape[0]
    test_size = int(test_frac * m)
    perm = np.random.permutation(m)

    test = data.iloc[perm[:test_size], :]
    train = data.iloc[perm[test_size:], :]

    return train, test


def split_train_dev_test(data, dev_frac, test_frac, seed=None):
    if seed:
        np.random.seed(seed)

    m  = data.shape[0]
    dev_size = int(dev_frac * m)
    test_size = int(test_frac * m)
    perm = np.random.permutation(m)

    dev = data.iloc[perm[:dev_size], :]
    test = data.iloc[perm[dev_size:dev_size + test_size], :]
    train = data.iloc[perm[dev_size + test_size:], :]

    return train, dev, test


def rearrange_by_epigenetic_marker(df):
    new_cols = []
    for m in range(8):
        idx = [i for i in range(m, 1016, 8)]
        new_cols.extend(idx)
    return df.iloc[:, new_cols]


def downsample_negatives(train, p):

    train_negs = train[train.Label == 0]
    train_pos = train[train.Label == 1]

    train_downsample = resample(train_negs,
                                replace=False,
                                n_samples=int(p * train_negs.shape[0]),
                                random_state=111)

    train_balanced = pd.concat([train_downsample, train_pos])
    return train_balanced.sample(frac=1, axis=0)


def upsample_positives(train, scale):

    train_negs = train[train.Label == 0]
    train_pos = train[train.Label == 1]

    train_upsample = resample(train_pos,
                              replace=True,
                              n_samples=scale * train_pos.shape[0],
                              random_state=111)

    train_balanced = pd.concat([train_negs, train_upsample])
    return train_balanced.sample(frac=1, axis=0)
