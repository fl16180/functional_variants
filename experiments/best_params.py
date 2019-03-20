def get_baseline_params():
    params = {}
    params['n_hidden'] = 2
    params['n1'] = 676
    params['n2'] = 72

    params['use_drop'] = 0

    params['lambda'] = 1.3e-6
    params['learning_rate'] = 0.00008
    params['epochs'] = 25
    params['batch_size'] = 256
    params['weight_1'] = 3.
    params['hidden_activation'] = 'relu'

    params['use_vat'] = 0
    params['use_pc'] = 0
    return params


# use this setting as a hparam example
# def get_dense_params():
#     params = {}
#     params['n_hidden'] = 2
#     params['n1'] = 300
#     params['n2'] = 100
#     params['use_drop'] = 1
#     params['drop1'] = 0.65
#     params['drop2'] = 0.5
#     params['lambda'] = 0
#     params['learning_rate'] = 0.0001
#     params['epochs'] = 20
#     params['batch_size'] = 256
#     params['weight_1'] = 3.
#     params['hidden_activation'] = 'sigmoid'
#     params['use_vat'] = 0
#
#     return params


# def get_test_params():
#     params = {}
#     params['n_hidden'] = 2
#     params['n1'] = 200
#     params['n2'] = 100
#     params['use_drop'] = 1
#     params['drop1'] = 0.024
#     params['drop2'] = 0.1
#     params['lambda'] = 0.00048
#     params['learning_rate'] = 0.00008
#     params['epochs'] = 25
#     params['batch_size'] = 256
#     params['weight_1'] = 3.
#     params['hidden_activation'] = 'sigmoid'
#     params['use_vat'] = 0
#     params['eps'] = 1
#     params['xi'] = 10
#     params['ip'] = 1
#
#     return params


def get_dense_params():
    params = {}
    params['n_hidden'] = 2
    params['n1'] = 400
    params['n2'] = 250

    params['use_drop'] = 1
    params['drop1'] = 0.47
    params['drop2'] = 0.6

    params['lambda'] = 1.8e-6
    params['learning_rate'] = 0.00017
    params['epochs'] = 25
    params['batch_size'] = 256
    params['weight_1'] = 1.
    params['hidden_activation'] = 'sigmoid'

    params['use_vat'] = 0
    params['eps'] = 1
    params['xi'] = 10
    params['ip'] = 1

    params['use_pc'] = 0

    return params


def get_pc_params():
    params = {}
    params['n_hidden'] = 2
    params['n1'] = 250
    params['n2'] = 180

    params['use_drop'] = 1
    params['pc_drop'] = 0
    params['drop1'] = 0.48
    params['drop2'] = 0.6

    params['lambda'] = 0
    params['learning_rate'] = 0.00027
    params['epochs'] = 25
    params['batch_size'] = 256
    params['weight_1'] = 3.
    params['hidden_activation'] = 'sigmoid'

    params['use_vat'] = 0
    params['eps'] = 1
    params['xi'] = 10
    params['ip'] = 1

    params['use_pc'] = 1
    params['npc'] = 100

    return params


def get_vat_params():
    params = {}
    params['n_hidden'] = 2
    params['n1'] = 400
    params['n2'] = 250

    params['use_drop'] = 1
    params['drop1'] = 0.47
    params['drop2'] = 0.6

    params['lambda'] = 6.74e-08
    params['learning_rate'] = 0.00017
    params['epochs'] = 25
    params['batch_size'] = 256
    params['weight_1'] = 1.
    params['hidden_activation'] = 'sigmoid'

    params['use_vat'] = 1
    params['eps'] = 0.04
    params['xi'] = 10
    params['ip'] = 1

    params['use_pc'] = 0
    return params


def get_seq_params():
    params = {}

    params['nf1'] = 50
    params['kernel1'] = 3
    params['n1'] = 100

    params['use_drop'] = 0
    params['lambda'] = 0
    params['learning_rate'] = 0.001
    params['epochs'] = 25
    params['batch_size'] = 64
    params['weight_1'] = 1.
    params['hidden_activation'] = 'relu'

    params['use_vat'] = 0
    params['eps'] = 1
    params['xi'] = 10
    params['ip'] = 1

    params['use_pc'] = 0
    return params
