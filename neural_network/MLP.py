from neural_network.InputNeuron import InputNeuron
from neural_network.HiddenNeuron import HiddenNeuron
from neural_network.OutputNeuron import OutputNeuron
import activation_functions.LogisticFunction as LOG
import activation_functions.StepFunction as STEP
import activation_functions.TanHFunction as TAN

import numpy as np

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
        '''
        :param input_neurons: Number of neurons for input layer.
        :param hidden_neurons: Number of neurons for hidden layer. Only one hidden layer is considered.
        :param output_neurons: Number of neurons for output layer.
        :param eta_min: Smaller value eta can reach with eta's decay.
        :param eta_max: Higher value that eta can reach with the decay of eta.
        :param eta_decay_lim: Percentage of epoch to limit of decay of the eta. Value range is [0, 1).
        :param epoch: Number of training iterations.
        :param hidden_act_func: {'default', 'logistic', 'tanh'}. default 'default'.
        \nType of activation function for hidden layer.
        :param output_act_func: {'default', 'logistic', 'tanh'}. default 'default'.
        \nType of activation function for output layer.
        '''
        self._input_neurons = input_neurons
        self._hidden_neurons = hidden_neurons + 1
        self._output_neurons = output_neurons
        self._eta_min = eta_min
        self._eta_max = eta_max
        self._eta_decay_lim = eta_decay_lim * epoch
        self._epoch = epoch
        self._hidden_act_func = hidden_act_func
        self._output_act_func = output_act_func

        # List of every error per epoch
        self.errors = list()

    def init_neurons(self):
        '''
        This function initialize all neurons with all parameters passed into constructor of this class.
        :return: nothing
        '''

        if self._hidden_act_func == 'default':
            self._hidden_act_func = STEP.StepFunction()
        elif self._hidden_act_func == 'logistic':
            self._hidden_act_func = LOG.LogisticFunction()
        elif self._hidden_act_func == 'tanh':
            self._hidden_act_func = TAN.TanHFunction()
        else:
            print("Incorrect parameter used for hidden activation function. Exiting...")
            exit(0)

        if self._output_act_func == 'default':
            self._output_act_func = STEP.StepFunction()
        elif self._output_act_func == 'logistic':
            self._output_act_func = LOG.LogisticFunction()
        elif self._output_act_func == 'tanh':
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


    def predict(self, X, y):
        pass

    def feedforward(self, xi):
        '''
        This function will feed the first layer, then, first layer\n
        feed the second layer, then, second layer feed the third layer.\n
        Then, all neurons will be activated.
        :param xi: This is an element of sample.
        :return: returns the computation for  all activations functions elements.
        '''
        # Active input neurons
        X = list()
        for i in range(self._input_neurons):
            self.input_neurons_layer[i].receive_impulse(xij=xi[i])
            X.append(self.input_neurons_layer[i].get_value())
            # print("InputNeuron[%d]._xij = %d" % (i, self.input_neurons_layer[i]._xij))
        X = np.array(X)

        # Active hidden neurons
        H = list()
        H_derivative = list()
        for j in range(self._hidden_neurons):
            self.hidden_neurons_layer[j].activation(X=X)
            H.append(self.hidden_neurons_layer[j].get_h())
            H_derivative.append(self.hidden_neurons_layer[j].get_h_derivative())

        # H = np.array(H)
        # H_derivative = np.array(H_derivative)

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
        '''
        :param activation_elements:
        :param eta:
        :param target:
        :return:
        '''
        # print(activation_elements)
        H, H_derivative, Y, Y_derivative = activation_elements
        # Start with output layer. In this layer, the vector weights is called m
        output_error = list()
        for j in range(self._output_neurons):
            _e_j = (target[j] - Y[j])
            output_error.append(_e_j) # Vector with all output neurons errors
            update = eta * _e_j * np.array(H) * Y_derivative[j]
            self.output_neurons_layer[j].update_weight(update=update)

        # Finish with weights of hidden layer. In this layer, the weights vector is called w
        for i in range(self._hidden_neurons):
            ei = 0
            for j in range(self._output_neurons):
                    ei += output_error[j] * Y_derivative[j] * self.output_neurons_layer[j].get_mj(p=i)
            update = eta * ei * H_derivative[i] * X
            self.hidden_neurons_layer[i].update_weight(update=update)

    @staticmethod
    def eta_decay(self, current_epoch, current_eta):
        '''
        :param self: MLP object with max, min eta and eta decay value.
        :param current_epoch: Current epoch running in fit.
        :param current_eta: Current eta running in fit.
        :return: eta updated.
        '''
        if current_epoch <= self._eta_decay_lim:
            return self._eta_max * pow((self._eta_min / self._eta_max),
                                       (current_epoch / self._eta_decay_lim))
        return current_eta
