from functools import reduce
import numpy as np

from keras import regularizers, optimizers
from keras.layers import Input, Dense, Dropout, Activation
from keras.layers import Lambda, Concatenate
from keras.models import Model
from keras.callbacks import Callback

from keras.utils.generic_utils import to_list
from keras.utils import np_utils
from keras import backend as K

from utils.metrics import f1, auroc, metric_report, avgPR


class ClfMetrics(Callback):
    ''' Keras callback to define validation metric. Here I use F1, AUPR, and AUROC.
    These metrics are cleanly defined over the entire validation set which I use, but not
    over sequences of batches, so Keras requires a callback to compute them.  '''
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_aupr = []
        self.val_auroc = []

    def on_epoch_end(self, epoch, logs={}):
        probs = np.array(self.model.predict(self.validation_data[0])).flatten()
        val_predict = (probs > 0.5).astype(float)
        val_targ = np.array(self.validation_data[1]).flatten()

        _val_f1 = f1(val_targ, val_predict)
        _val_aupr = avgPR(val_targ, probs)
        _val_auroc = auroc(val_targ, probs)

        self.val_f1s.append(_val_f1)
        self.val_aupr.append(_val_aupr)
        self.val_auroc.append(_val_auroc)
        print(" — val_f1: %f" %(_val_f1))
        print(" — val_aupr: %f" %(_val_aupr))
        print(" — val_auroc: %f" %(_val_auroc))
        return


class BaseModel:
    model = None

    def __init__(self, input_shape, params):
        self.input_shape = input_shape
        self.params = params
        self.vat = True if self.params['use_vat'] == 1 else False
        self.dropout = True if self.params['use_drop'] == 1 else False

    def build(self):
        input_layer = Input(self.input_shape)
        output_layer = self.network(input_layer)
        if self.vat:
            eps = self.params['eps']
            xi = self.params['xi']
            ip = self.params['ip']
            self.model = VATModel(input_layer, output_layer).setup_vat_loss(eps, xi, ip)
        else:
            self.model = Model(input_layer, output_layer)
        return self

    def network(self, input_layer):
        n_hidden = self.params['n_hidden']
        n1 = self.params['n1']
        activation = self.params['hidden_activation']
        lmbd = self.params['lambda']

        X = Dense(n1, activation=activation, kernel_regularizer=regularizers.l2(lmbd))(input_layer)
        if self.dropout:
            X = Dropout(rate=self.params['drop1'])(X)

        for l in range(n_hidden-1):
            X = Dense(self.params['n{0}'.format(l+1)],
                        activation=activation,
                        kernel_regularizer=regularizers.l2(lmbd))(X)

            if self.dropout:
                X = Dropout(rate=self.params['drop{0}'.format(l+1)])(X)
        X = Dense(1, activation='sigmoid')(X)

        return X

    def train(self, X_train, y_train, X_dev, y_dev):

        lr = self.params['learning_rate']
        batch_size = self.params['batch_size']
        epochs = self.params['epochs']
        weight_1 = self.params['weight_1'] if 'weight_1' in self.params else 1.

        metrics = ClfMetrics()
        callbacks = [metrics]
        class_weight = {0: 1., 1: weight_1}

        adam = optimizers.Adam(lr=lr, beta_1=0.9)
        self.model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

        np.random.seed(230)
        self.model.fit(X_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(X_dev, y_dev),
                            shuffle=True,
                            callbacks=callbacks,
                            class_weight=class_weight,
                            verbose=1)

    def predict(self, X):
        return self.model.predict(X)


class PartialModel(BaseModel):
    ''' partially connected hidden layer grouping epigenetic markers over tissues of a given type
    '''
    def network(self, input_layer):

        npc = self.params['npc']
        activation = self.params['hidden_activation']

        nodes = []
        for m in range(8):
            z = Lambda(lambda x: x[:, m*127:(m+1)*127], output_shape=(127,))(input_layer)
            z = Dense(self.params['npc'], activation=activation)(z)
            if self.params['pc_drop'] > 0:
                z = Dropout(self.params['pc_drop'])(z)
            nodes.append(z)

        X = Concatenate()(nodes)
        X = Dense(self.params['n1'], activation=activation)(X)
        if self.dropout:
            X = Dropout(rate=self.params['drop1'])(X)

        if self.params['n_hidden'] == 3:
            X = Dense(self.params['n2'], activation=activation)(X)
            if self.dropout:
                X = Dropout(rate=self.params['drop2'])(X)

        X = Dense(1, activation='sigmoid')(X)

        return X


class VATModel(Model):
    ''' implementation adapted from
    https://gist.github.com/mokemokechicken/2658d036c717a3ed07064aa79a59c82d'''
    _vat_loss = None

    def setup_vat_loss(self, eps=1, xi=10, ip=1):
        self._vat_loss = self.vat_loss(eps, xi, ip)
        return self

    @property
    def losses(self):
        losses = super(self.__class__, self).losses
        if self._vat_loss is not None:
            losses += [self._vat_loss]
        return losses

    def vat_loss(self, eps, xi, ip):
        normal_outputs = [K.stop_gradient(x) for x in to_list(self.outputs)]
        d_list = [K.random_normal(K.shape(x)) for x in self.inputs]

        for _ in range(ip):
            new_inputs = [x + self.normalize_vector(d)*xi for (x, d) in zip(self.inputs, d_list)]
            new_outputs = to_list(self.call(new_inputs))
            klds = [K.sum(self.kld(normal, new)) for normal, new in zip(normal_outputs, new_outputs)]
            kld = reduce(lambda t, x: t+x, klds, 0)
            d_list = [K.stop_gradient(d) for d in K.gradients(kld, d_list)]

        new_inputs = [x + self.normalize_vector(d) * eps for (x, d) in zip(self.inputs, d_list)]
        y_perturbations = to_list(self.call(new_inputs))
        klds = [K.mean(self.kld(normal, new)) for normal, new in zip(normal_outputs, y_perturbations)]
        kld = reduce(lambda t, x: t + x, klds, 0)
        return kld

    @staticmethod
    def normalize_vector(x):
        z = K.sum(K.batch_flatten(K.square(x)), axis=1)
        while K.ndim(z) < K.ndim(x):
            z = K.expand_dims(z, axis=-1)
        return x / (K.sqrt(z) + K.epsilon())

    @staticmethod
    def kld(p, q):
        v = p * (K.log(p + K.epsilon()) - K.log(q + K.epsilon()))
        return K.sum(K.batch_flatten(v), axis=1, keepdims=True)
