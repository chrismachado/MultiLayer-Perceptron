import numpy as np


class OutputNeuron(object):
    def __init__(self, activation_func,
                 hidden_neurons_size,
                 output_neurons_size):
        self._activation_func = activation_func
        # self._m = np.ones(shape=(hidden_neurons_size), dtype=float)
        self._m = np.random.randn(hidden_neurons_size) * np.sqrt(2 / (hidden_neurons_size + output_neurons_size) )
        self._uj = None
        self._y = None
        self._y_derivative = None

    def activation(self, H):
        self._uj = np.dot(self._m, H) #mTh
        self._y = self._activation_func.function(self._uj)
        self._y_derivative = self._activation_func.derivative(self._uj)

    def get_y(self):
        return self._y

    def get_y_derivative(self):
        return self._y_derivative

    def get_m(self):
        return self._m

    def get_mj(self, p):
        return self._m[p]

    def set_m(self, m):
        self._m = m

    def update_weight(self, update):
        self._m += update
