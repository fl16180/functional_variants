import pandas as pd
import numpy as np

from model.data_loaders import load_train_set, load_test_set, load_benchmark
from utils.metrics import auroc, bin_probs, metric_report, avgPR


def get_benchmarks_for_slice(data_slice, benchmarks):
    ''' Pulls benchmark scores for non-coding variant functionality for a given set of
        genomic positions.

        Input:
            data_slice: Dataframe requiring columns chr and pos
            benchmarks: Official benchmark dataframe from load_benchmark function.
    '''
    return pd.merge(data_slice[['chr','pos']], benchmarks)


def logistic_benchmark(train, dev):
    ''' Fits regularized logistic regression benchmark on input train and dev dataframes.
    '''
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler

    X_train = train.iloc[:, 4:]
    y_train = train.Label.values
    X_dev = dev.iloc[:, 4:]
    y_dev = dev.Label.values

    X_train = np.log(X_train)
    X_dev = np.log(X_dev)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_dev = scaler.transform(X_dev)

    clf = LogisticRegressionCV(Cs=5, cv=3, max_iter=500)
    clf.fit(X_train, y_train)

    train_pred = clf.predict(X_train)
    probs_train = clf.predict_proba(X_train)

    dev_pred = clf.predict(X_dev)
    probs_dev = clf.predict_proba(X_dev)

    results = {}
    train_avgpr = avgPR(y_train, probs_train[:,1])
    dev_avgpr = avgPR(y_dev, probs_dev[:,1])
    train_auroc = auroc(y_train, probs_train[:,1])
    dev_auroc = auroc(y_dev, probs_dev[:,1])

    results['AUPR_train'] = train_avgpr
    results['AUROC_train'] = train_auroc
    results['AUPR_dev'] = dev_avgpr
    results['AUROC_dev'] = dev_auroc
    results['AUPR_bench'] = 0
    results['AUROC_bench'] = 0

    return results, clf


def official_benchmark_splits(train, dev, test, benchmark='GNET'):
    ''' Loads benchmark predictions corresponding to the given train and dev dataframes
        and reports metrics on the benchmark model.
    '''
    bench = load_benchmark(dataset='E116')
    train_bench = get_benchmarks_for_slice(train, bench)
    dev_bench = get_benchmarks_for_slice(dev, bench)
    test_bench = get_benchmarks_for_slice(test, bench)

    metric_report(train_bench['Label'], bin_probs(train_bench[benchmark]))
    metric_report(dev_bench['Label'], bin_probs(dev_bench[benchmark]))
    metric_report(test_bench['Label'], bin_probs(test_bench[benchmark]))

    print(auroc(train_bench['Label'], train_bench[benchmark]))
    print(auroc(dev_bench['Label'], dev_bench[benchmark]))
    print(auroc(test_bench['Label'], test_bench[benchmark]))

    print(avgPR(train_bench['Label'], train_bench[benchmark]))
    print(avgPR(dev_bench['Label'], dev_bench[benchmark]))
    print(avgPR(test_bench['Label'], test_bench[benchmark]))


def get_benchmark_score(data, benchmark='GNET'):
    bdf = load_benchmark(dataset='E116')
    bench = get_benchmarks_for_slice(data, bdf)

    bench_avgpr = avgPR(bench['Label'], bench[benchmark])
    bench_auroc = auroc(bench['Label'], bench[benchmark])

    return bench_avgpr, bench_auroc


def main():
    # train, dev = load_train_dev(dataset='E116')
    # test = load_test(dataset='E116')
    # logistic_benchmark(train, dev)

    official_benchmark(train, dev, test)


if __name__ == '__main__':
    main()
