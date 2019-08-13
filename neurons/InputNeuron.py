import numpy as np


class InputNeuron(object):

    def __init__(self):
        self._xij = None

    def receive_impulse(self, xij):
        '''
        Receive the specific value of the element of the samples, then change the value of
        \nthis neuron.
        :param xij: Specific value of the element of the samples.
        :return: nothing.
        '''
        self._xij = xij

    def get_value(self):
        '''
        :return: return the value of this neuron
        '''
        return self._xij
