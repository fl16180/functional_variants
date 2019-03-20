import pandas as pd
import numpy as np
from os.path import join
from sklearn.preprocessing import StandardScaler

from config import Config as cfg
from model.data_loaders import load_train_set, load_test_set, load_seq_train_set, load_seq_test_set
from model.models import BaseModel, PartialModel
from model.seq_models import SeqModel, MergeModel, encode_seq
from utils.metrics import auroc, avgPR
from utils.data_utils import split_train_test, rearrange_by_epigenetic_marker, downsample_negatives
from utils.param_utils import compute_averages, update_results, summarize_results
from utils.benchmarks import get_benchmark_score, logistic_benchmark
from experiments.best_params import *


def splits_single_model(data, params, mod):

    trials = {}
    for i, seed in enumerate([100, 200, 300]):
        train, dev = split_train_test(data, test_frac=0.2, seed=seed)

        if mod == 'mpra':
            run_results, _ = train_model(train, dev, params)
        elif mod == 'seq':
            run_results, _ = train_seq_model(train, dev, params)
        elif mod == 'lr-bench':
            run_results, _ = logistic_benchmark(train, dev)

        trials = update_results(trials, run_results, i+1)
    trials = compute_averages(trials)

    return trials


def train_model(train, dev, params, return_probs=False):

    X_train = train.drop(['chr', 'pos', 'rs', 'Label'], axis=1)
    y_train = train.Label.values
    X_dev = dev.drop(['chr', 'pos', 'rs', 'Label'], axis=1)
    y_dev = dev.Label.values

    X_train = rearrange_by_epigenetic_marker(X_train)
    X_dev = rearrange_by_epigenetic_marker(X_dev)

    # X_train = np.log(X_train)
    # X_dev = np.log(X_dev)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_dev = scaler.transform(X_dev)

    X_shape = (X_train.shape[1],)
    if params['use_pc'] == 1:
        mod = PartialModel(input_shape=X_shape, params=params).build()
    else:
        mod = BaseModel(input_shape=X_shape, params=params).build()


    mod.train(X_train, y_train, X_dev, y_dev)

    probs_train = mod.predict(X_train)
    probs_dev = mod.predict(X_dev)

    results = {}
    train_avgpr = avgPR(y_train, probs_train)
    dev_avgpr = avgPR(y_dev, probs_dev)
    train_auroc = auroc(y_train, probs_train)
    dev_auroc = auroc(y_dev, probs_dev)
    bench_avgpr, bench_auroc = get_benchmark_score(dev)

    results['AUPR_train'] = train_avgpr
    results['AUROC_train'] = train_auroc
    results['AUPR_dev'] = dev_avgpr
    results['AUROC_dev'] = dev_auroc
    results['AUPR_bench'] = bench_avgpr
    results['AUROC_bench'] = bench_auroc

    if return_probs:
        return probs_dev

    return results, mod


def train_seq_model(train, dev, params):

    train = downsample_negatives(train, p=0.25)
    print(train.shape)
    X_train = train['seq']
    y_train = train.Label.values
    X_dev = dev['seq']
    y_dev = dev.Label.values

    X_train = encode_seq(X_train)
    X_dev = encode_seq(X_dev)

    X_shape = (19,4)
    mod = SeqModel(input_shape=X_shape, params=params).build()

    mod.train(X_train, y_train, X_dev, y_dev)

    probs_train = mod.predict(X_train)
    probs_dev = mod.predict(X_dev)

    results = {}
    train_avgpr = avgPR(y_train, probs_train)
    dev_avgpr = avgPR(y_dev, probs_dev)
    train_auroc = auroc(y_train, probs_train)
    dev_auroc = auroc(y_dev, probs_dev)
    bench_avgpr, bench_auroc = get_benchmark_score(dev)

    results['AUPR_train'] = train_avgpr
    results['AUROC_train'] = train_auroc
    results['AUPR_dev'] = dev_avgpr
    results['AUROC_dev'] = dev_auroc
    results['AUPR_bench'] = bench_avgpr
    results['AUROC_bench'] = bench_auroc

    return results, mod


def evaluate_model(train, test, params, mod):
    if mod == 'mpra':
        results, mod = train_model(train, test, params)
    elif mod == 'seq':
        results, mod = train_seq_model(train, test, params)
    elif mod == 'lr-bench':
        results, mod = logistic_benchmark(train, test)

    summarize_results(results)


if __name__ == '__main__':


    # train models on iterated train-dev splits
    data = load_train_set(dataset='E116')

    print('------ Iterated train-dev performances ------')
    # print('Baseline: ')
    # params = get_baseline_params()
    # trials = splits_single_model(data, params, mod='mpra')
    # summarize_results(trials)
    #
    # print('\nDense: ')
    # params = get_dense_params()
    # trials = splits_single_model(data, params, mod='mpra')
    # summarize_results(trials)
    #
    # print('\nPartially Connected: ')
    # params = get_pc_params()
    # trials = splits_single_model(data, params, mod='mpra')
    # summarize_results(trials)
    #
    # print('\nVAT: ')
    # params = get_vat_params()
    # trials = splits_single_model(data, params, mod='mpra')
    # summarize_results(trials)

    params = get_baseline_params()
    trials = splits_single_model(data, params, mod='lr-bench')
    summarize_results(trials)


    # retrain models on entire train set and evaluate on test set
    test = load_test_set(dataset='E116')

    print('------ Evaluate train-test performances ------')
    # print('Baseline: ')
    # # params = get_baseline_params()
    # # evaluate_model(data, test, params, mod='mpra')
    #
    # print('\nDense: ')
    # params = get_dense_params()
    # evaluate_model(data, test, params, mod='mpra')
    #
    # print('\nPartially Connected: ')
    # params = get_pc_params()
    # evaluate_model(data, test, params, mod='mpra')
    #
    # print('\nVAT: ')
    # params = get_vat_params()
    # evaluate_model(data, test, params, mod='mpra')

    params = get_baseline_params()
    evaluate_model(data, test, params, mod='lr-bench')

    #
    # # save predictions for test set with dense net
    # params = get_dense_params()
    # X_probs = train_model(data, test, params, return_probs=True)
    # test['Score'] = X_probs
    # test.to_csv(join(cfg.OUTPUT_DIR, 'scores.csv'), index=False)
