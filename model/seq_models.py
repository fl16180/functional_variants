import numpy as np

from keras import regularizers, optimizers
from keras.layers import Input, Dense, Dropout, Activation
from keras.layers import Flatten, Concatenate
from keras.models import Model
from keras.layers import Conv1D, MaxPooling1D, SeparableConv1D
from keras import backend as K
from keras.utils import to_categorical

from model.models import ClfMetrics, BaseModel, VATModel
from utils.metrics import f1, auroc, metric_report, avgPR


def seq_to_labels(dna_str):
    BASES = {'A': 0, 'T': 1, 'C': 2, 'G': 3}

    labels = [BASES[x] for x in dna_str]
    return labels


def encode_seq(data):
    data = to_categorical(np.vstack(data.map(seq_to_labels)))
    return data


class SeqModel(BaseModel):

    def network(self, input_layer):
        nf1 = self.params['nf1']
        ker1 = self.params['kernel1']
        n1 = self.params['n1']
        activation = self.params['hidden_activation']

        X = SeparableConv1D(filters=nf1, kernel_size=ker1, activation=activation)(input_layer)

        X = Flatten()(X)
        X = Dense(n1, activation=activation)(X)
        if self.dropout:
            X = Dropout(rate=self.params['drop1'])(X)
        X = Dense(1, activation='sigmoid')(X)

        return X


class MergeModel(BaseModel):

    def network(self, input_layer):
        raise NotImplementedError
