from neurons.InputNeuron import InputNeuron
from neurons.HiddenNeuron import HiddenNeuron
from neurons.OutputNeuron import OutputNeuron
from sklearn.metrics import mean_squared_error

from utilities.VectorUtilities import VectorUtilities

import activation_functions.LogisticFunction as LOG
import activation_functions.StepFunction as STEP
import activation_functions.TanHFunction as TAN
import activation_functions.RegressionFunction as REG

import numpy as np
import copy as cp


class ELM(object):
    def __init__(self,
                 input_neurons,
                 hidden_neurons,
                 output_neurons,
                 hidden_act_func='default',
                 output_act_func='default'):

        self._input_neurons = input_neurons
        self._hidden_neurons = hidden_neurons
        self._output_neurons = output_neurons
        self._hidden_act_func_ = hidden_act_func
        self._output_act_func_ = output_act_func

        self._epoch = None

    def init_neurons(self):
        if self._hidden_act_func_ == 'default':
            self._hidden_act_func = STEP.StepFunction()
        elif self._hidden_act_func_ == 'logistic':
            self._hidden_act_func = LOG.LogisticFunction()
        elif self._hidden_act_func_ == 'tanh':
            self._hidden_act_func = TAN.TanHFunction()
        else:
            print("Incorrect parameter used for hidden activation function. Exiting...")
            exit(0)

        if self._output_act_func_ == 'default':
            self._output_act_func = STEP.StepFunction()
        elif self._output_act_func_ == 'logistic':
            self._output_act_func = LOG.LogisticFunction()
        elif self._output_act_func_ == 'tanh':
            self._output_act_func = TAN.TanHFunction()
        elif self._output_act_func_ == 'regression':
            self._output_act_func = REG.RegressionFunction()
        else:
            print("Incorrect parameter used for output activation function. Exiting.")
            exit(0)

        # Start input layer neurons
        self.input_neurons_layer = list()
        for i in range(self._input_neurons):
            self.input_neurons_layer.append(InputNeuron())

        # Start hidden layer neurons
        self.hidden_neurons_layer = list()
        for j in range(self._hidden_neurons):
            self.hidden_neurons_layer.append(HiddenNeuron(activation_func=self._hidden_act_func,
                                                          input_neurons_size=self._input_neurons,
                                                          hidden_neurons_size=self._hidden_neurons))

        # Start output layer neurons
        self.output_neurons_layer = list()
        for k in range(self._output_neurons):
            self.output_neurons_layer.append(OutputNeuron(activation_func=self._output_act_func,
                                                          hidden_neurons_size=self._hidden_neurons,
                                                          output_neurons_size=self._output_neurons))

    def fit(self, X, y):
        self.init_neurons()
        X = np.c_[-np.ones(X.shape[0]), X]
        self.feedforward(X, y)

    def feedforward(self, X, y):
        W, M = self.join_weights()

        #Training
        U = X.dot(W.T)

        H = self._hidden_act_func.function(U) # XTW
        H = np.c_[-np.ones(H.shape[0]), H] # bias added
        Ht = np.transpose(H)

        M = np.dot(np.linalg.pinv(np.dot(Ht, H)), np.dot(Ht, y))

        for i in range(self._hidden_neurons):
            self.hidden_neurons_layer[i].set_w(W[i])

        for j in range(self._output_neurons):
            self.output_neurons_layer[j].set_m(M.T[j])

    def predict(self, X, y):
        if self._output_act_func_ == 'regression':
            return self.estimate(X, y)
        elif self._output_act_func_ == 'logistic' or self._output_act_func_ == 'tanh':
            return self.classify(X, y)

    def classify(self, X, y):
        hitrate = 0
        for xi, target in zip(X, y):
            Y = self.feedfoward_output(xi)
            y_obtained = self.around(Y)

            if np.array_equal(y_obtained.astype(int), target):
                hitrate += 1

        return hitrate / X.shape[0]

    def estimate(self, X, y):
        y_obtained = list()
        for x in X:
            Y = self.feedfoward_output(x)
            y_obtained.append(Y)

        return mean_squared_error(y, y_obtained)

    def predictions(self, X):
        predictions = list()

        for xi in X:
            xi = np.insert(xi, 0, -1)
            H = list()
            for i in range(self._hidden_neurons):
                self.hidden_neurons_layer[i].activation(xi)
                H.append(self.hidden_neurons_layer[i].get_h())

            H = np.insert(H, 0, -1)
            for j in range(self._output_neurons):
                self.output_neurons_layer[j].activation(H)
                predictions.append((self.output_neurons_layer[0].get_y()))

        return predictions

    def around(self, prediction_):
        prediction = cp.deepcopy(prediction_)
        _max = max(prediction)
        for i in range(len(prediction)):
            if prediction[i] == _max:
                prediction[i] = 1
            else:
                if self._output_act_func_ == 'tanh':
                    prediction[i] = -1
                else:
                    prediction[i] = 0
        return np.array(prediction, dtype=float)

    def predict_1D(self, X):
        y = list()
        for xi in X:
            Y = self.feedfoward_output(xi)
            y.append(self.around(Y))
        return np.array(y)

    def boundary_decision(self, X):
        y = list()
        for xi in X:
            Y = self.feedfoward_output(xi)
            y_obtained = self.around(Y)
            if np.array_equal(y_obtained.astype(int), [0, 1]):
                y.append(0)
            elif np.array_equal(y_obtained.astype(int), [1, 0]):
                y.append(1)
        return np.array(y)

    def join_weights(self):
        W = list()
        M = list()
        for i in range(self._hidden_neurons):
            W.append(self.hidden_neurons_layer[i].get_w().T)

        for j in range(self._output_neurons):
            M.append(self.output_neurons_layer[j].get_m().T)

        return np.asarray(W), np.asarray(M)

    def feedfoward_output(self, x):
        x = np.insert(x, 0, -1)
        H = list()
        for i in range(self._hidden_neurons):
            self.hidden_neurons_layer[i].activation(x)
            H.append(self.hidden_neurons_layer[i].get_h())

        H = np.insert(H, 0, -1)
        Y = list()

        for j in range(self._output_neurons):
            self.output_neurons_layer[j].activation(H)
            Y.append(self.output_neurons_layer[j].get_y())
        return np.asarray(Y)