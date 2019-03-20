

def compute_averages(trials):

    trials['AUPR_train'] = (trials['AUPR_train_1'] + trials['AUPR_train_2'] + trials['AUPR_train_3']) / 3
    trials['AUPR_dev'] = (trials['AUPR_dev_1'] + trials['AUPR_dev_2'] + trials['AUPR_dev_3']) / 3
    trials['AUPR_bench'] = (trials['AUPR_bench_1'] + trials['AUPR_bench_2'] + trials['AUPR_bench_3']) / 3
    trials['AUROC_train'] = (trials['AUROC_train_1'] + trials['AUROC_train_2'] + trials['AUROC_train_3']) / 3
    trials['AUROC_dev'] = (trials['AUROC_dev_1'] + trials['AUROC_dev_2'] + trials['AUROC_dev_3']) / 3
    trials['AUROC_bench'] = (trials['AUROC_bench_1'] + trials['AUROC_bench_2'] + trials['AUROC_bench_3']) / 3

    return trials


def update_results(trials, results, iter):
    for k in results:
        trials[k + '_{0}'.format(iter)] = results[k]
    return trials


def summarize_results(results):
    print('\n---------- SUMMARY ----------')
    if 'AUPR_train' in results:
        print('Training AUPR: {0}'.format(results['AUPR_train']))
    if 'AUROC_train' in results:
        print('Training AUROC: {0}'.format(results['AUROC_train']))

    if 'AUPR_dev' in results:
        print('Test AUPR: {0}'.format(results['AUPR_dev']))
    if 'AUROC_dev' in results:
        print('Test AUROC: {0}'.format(results['AUROC_dev']))

    if 'AUPR_bench' in results:
        print('Test benchmark AUPR: {0}'.format(results['AUPR_bench']))
    if 'AUROC_dev' in results:
        print('Test benchmark AUROC: {0}'.format(results['AUROC_bench']))
