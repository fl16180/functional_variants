import pandas as pd
import numpy as np
from os.path import join
from sklearn.preprocessing import StandardScaler

from config import Config as cfg
from model.data_loaders import load_train_set, load_test_set
from model.models import BaseModel, PartialModel
from utils.metrics import f1, auroc, metric_report, avgPR
from utils.data_utils import split_train_test, rearrange_by_epigenetic_marker
from utils.benchmarks import get_benchmark_score
from train_and_evaluate import splits_single_model, train_model, compute_averages, update_results


def get_random_params_base():
    params = {}
    params['n_hidden'] = np.random.choice([2,3])
    params['n1'] = np.random.randint(low=200, high=600)
    params['n2'] = np.random.randint(low=50, high=300)
    params['n3'] = np.random.randint(low=50, high=200)

    params['use_drop'] = 1
    params['drop1'] = np.random.uniform(low=0.0, high=0.8)
    params['drop2'] = np.random.uniform(low=0.0, high=0.7)

    params['lambda'] = 10 ** (- 5 * np.random.rand() - 3)
    params['learning_rate'] = np.random.uniform(low=0.00002, high=0.0002)
    params['epochs'] = 25
    params['batch_size'] = 256
    params['weight_1'] = 3.
    params['hidden_activation'] = 'sigmoid'

    params['use_vat'] = 0
    params['eps'] = 1
    params['xi'] = 10
    params['ip'] = 1

    params['use_pc'] = 0
    return params


def get_random_params_pc():
    params = {}
    params['n_hidden'] = np.random.choice([2,3])
    params['n1'] = np.random.randint(low=200, high=600)
    params['n2'] = np.random.randint(low=50, high=300)
    params['n3'] = np.random.randint(low=50, high=200)

    params['use_drop'] = 1
    params['drop1'] = np.random.uniform(low=0.0, high=0.7)
    params['drop2'] = np.random.uniform(low=0.0, high=0.7)

    params['lambda'] = 10 ** (- 5 * np.random.rand() - 3)
    params['learning_rate'] = np.random.uniform(low=0.00005, high=0.0005)
    params['epochs'] = 25
    params['batch_size'] = 256
    params['weight_1'] = 3.
    params['hidden_activation'] = 'sigmoid'

    params['use_vat'] = 0
    params['eps'] = 1
    params['xi'] = 10
    params['ip'] = 1

    params['use_pc'] = 1
    params['npc'] = np.random.randint(low=50, high=127)

    if np.random.rand() > 0.5:
        params['pc_drop'] = np.random.uniform(low=0.0, high=0.5)
    else:
        params['pc_drop'] = 0
    return params


def get_random_params_vat():
    params = {}
    params['n_hidden'] = 2
    params['n1'] = 400
    params['n2'] = 250

    params['use_drop'] = 1
    params['drop1'] = 0.47
    params['drop2'] = 0.6

    params['lambda'] = np.random.uniform(low=0.0, high=3.6e-6)
    params['learning_rate'] = 0.00017
    params['epochs'] = 25
    params['batch_size'] = 256
    params['weight_1'] = 1.
    params['hidden_activation'] = 'sigmoid'

    params['use_vat'] = 1
    params['eps'] = np.random.uniform(low=0.01, high=0.2)
    params['xi'] = np.random.uniform(low=3.0, high=10.0)
    params['ip'] = 1

    params['use_pc'] = 0
    return params


def execute_random_search(data, outfile, iterations, param_fn):
    ''' runs random search for user-specified number of iterations.
        Preinitializes random parameters and fits models on fixed seed
        train/dev splits using the splits_single_model function.

        data: input (training) dataset
        outfile: output csv
    '''
    all_results = pd.DataFrame()
    param_storage = []

    random_param_list = [param_fn() for _ in range(iterations)]
    for iter in range(iterations):
        print('Iteration {0}: '.format(iter))

        params = random_param_list[iter]
        trials = splits_single_model(data, params, mod='mpra')

        params.update(trials)
        param_storage.append(params)

        # periodically save results to file and flush param_storage
        if iter % 2 == 1:
            all_results = pd.concat([all_results, pd.DataFrame(param_storage)])
            all_results.to_csv(outfile, index=False)
            param_storage = []


if __name__ == '__main__':

    data = load_train_set(dataset='E116')
    outfile = join(cfg.OUTPUT_DIR, 'hparams_vat2.csv')

    execute_random_search(data, outfile=outfile, iterations=50, param_fn=get_random_params_vat)
