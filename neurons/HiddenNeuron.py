import numpy as np


class HiddenNeuron(object):
    def __init__(self,
                 activation_func,
                 input_neurons_size,
                 hidden_neurons_size):
        self._activation_func = activation_func
        self._w = np.zeros(shape=(input_neurons_size + 1), dtype=float)
        # self._w = np.random.randn(input_neurons_size + 1) * np.sqrt(2 / (input_neurons_size + hidden_neurons_size + 1) )
        self._ui = None
        self._h = None
        self._h_derivative = None

    def activation(self, X):
        self._ui = np.dot(self._w[1: ], X[1: ]) + self._w[0]*X[0] #wTx + teta
        self._h = self._activation_func.function(self._ui)
        self._h_derivative = self._activation_func.derivative(self._ui)

    def get_h(self):
        return self._h

    def get_h_derivative(self):
        return self._h_derivative

    def get_w(self):
        return self._w

    def get_wi(self, p):
        return self._w[p]

    def get_u(self):
        return self._ui

    def update_weight(self, update):
        self._w += update