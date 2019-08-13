from neural_network.InputNeuron import InputNeuron
from neural_network.HiddenNeuron import HiddenNeuron
from neural_network.OutputNeuron import OutputNeuron
import activation_functions.LogisticFunction as LOG
import activation_functions.StepFunction as STEP
import activation_functions.TanHFunction as TAN

import numpy as np
import copy as cp


class MLP(object):
    def __init__(self,
                 input_neurons,
                 hidden_neurons,
                 output_neurons,
                 eta_min=0.001,
                 eta_max=0.5,
                 eta_decay_lim=0.8,
                 epoch=50,
                 hidden_act_func='default',
                 output_act_func='default'):
        self._input_neurons = input_neurons
        self._hidden_neurons = hidden_neurons
        self._output_neurons = output_neurons
        self._eta_min = eta_min
        self._eta_max = eta_max
        self._eta_decay_lim = eta_decay_lim * epoch
        self._epoch = epoch
        self._hidden_act_func_ = hidden_act_func
        self._output_act_func_ = output_act_func

        # List of every error per epoch
        self.errors = list()

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
        current_eta = self._eta_max

        for current_epoch in range(self._epoch):
            self.shuffle_(X=X, y=y)
            for xi, target in zip(X, y):
                _activation_elements = self.feedforward(xi=xi)

                # Back propagation Algorithm
                self.backpropagation(activation_elements=_activation_elements,
                                     X=xi,
                                     eta=current_eta,
                                     target=target)

                current_eta = self.eta_decay(self=self,
                               current_epoch=current_epoch,
                               current_eta=current_eta)

    def feedforward(self, xi):
        # Active input neurons
        X = list()
        for i in range(self._input_neurons):
            self.input_neurons_layer[i].receive_impulse(xij=xi[i])
            X.append(self.input_neurons_layer[i].get_value())
        X = np.insert(X, 0, - 1)

        # Active hidden neurons
        H = list()
        H_derivative = list()
        for j in range(self._hidden_neurons):
            self.hidden_neurons_layer[j].activation(X=X)
            H.append(self.hidden_neurons_layer[j].get_h())
            H_derivative.append(self.hidden_neurons_layer[j].get_h_derivative())

        H = np.insert(H, 0, -1)

        # Active output neurons
        Y = list()
        Y_derivative = list()
        for k in range(self._output_neurons):
            self.output_neurons_layer[k].activation(H=H)
            Y.append(self.output_neurons_layer[k].get_y())
            Y_derivative.append(self.output_neurons_layer[k].get_y_derivative())

        # Y = np.array(Y)
        # Y_derivative = np.array(Y_derivative)
        return H, H_derivative, Y, Y_derivative

    def backpropagation(self, activation_elements, X, eta, target):
        H, H_derivative, Y, Y_derivative = activation_elements
        X = np.insert(X, 0, -1)

        # Start with output layer. In this layer, the vector weights is called m
        output_error = list()
        for j in range(self._output_neurons):
            _e_j = (target[j] - Y[j])
            output_error.append(_e_j) # Vector with all output neurons errors
            update = eta * _e_j * np.array(H) * Y_derivative[j]
            self.output_neurons_layer[j].update_weight(update=update)
        output_error = np.array(output_error)

        # Finish with weights of hidden layer. In this layer, the weights vector is called w
        for i in range(self._hidden_neurons):
            ei = 0
            for j in range(self._output_neurons):
                ei += output_error[j] * Y_derivative[j] * self.output_neurons_layer[j].get_m()[i]
            update = eta * ei * H_derivative[i] * np.array(X)
            self.hidden_neurons_layer[i].update_weight(update=update)

    def test(self, X, y):
        hitrate = 0
        for xi, target in zip(X, y):
            Y = self.predict(xi=xi)
            y_obtained = self.around(Y)

            if np.array_equal(y_obtained.astype(int), target):
                hitrate += 1
        return hitrate / X.shape[0]

    def predict(self, xi):
        Y = self.feedforward(xi=xi)[2]
        return Y

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


    @staticmethod
    def eta_decay(self, current_epoch, current_eta):
        if current_epoch <= self._eta_decay_lim:
            return self._eta_max * pow((self._eta_min / self._eta_max),
                                       (current_epoch / self._eta_decay_lim))
        return current_eta

    def shuffle_(self, X, y):
        state = np.random.get_state()
        np.random.shuffle(X)
        np.random.set_state(state)
        np.random.shuffle(y)