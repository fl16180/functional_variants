import os
import pandas as pd
import numpy as np

from config import Config as cfg


def load_train_set(dataset):
    ''' convenience function for loading processed train or test splits of a dataset.
        dataset: 'E116', 'E118', etc.
    '''
    train = pd.read_csv(os.path.join(cfg.TRAIN_DIR, '{0}_train.csv'.format(dataset)))
    return train


def load_test_set(dataset):
    test = pd.read_csv(os.path.join(cfg.TEST_DIR, '{0}_test.csv'.format(dataset)))
    return test


def load_seq_train_set(dataset):
    train_seq = pd.read_csv(os.path.join(cfg.TRAIN_DIR, '{0}_seq_train.csv'.format(dataset)))
    return train_seq


def load_seq_test_set(dataset):
    test_seq = pd.read_csv(os.path.join(cfg.TEST_DIR, '{0}_seq_test.csv'.format(dataset)))
    return test_seq


def load_benchmark(dataset):
    ''' loads benchmark table corresponding to the dataset
    '''
    bench = pd.read_csv(os.path.join(cfg.BENCH_DIR, '{0}_benchmarks.csv'.format(dataset)))
    return bench
