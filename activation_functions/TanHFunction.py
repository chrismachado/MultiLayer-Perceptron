import numpy as np


class TanHFunction(object):
    '''
    This class is responsible for tanh functions operations.
    '''

    @staticmethod
    def function(u):
        return (1.0 - np.exp(-u))/(1.0 + np.exp(-u))

    @staticmethod
    def derivative(u):
        y = TanHFunction.function(u)
        return 0.5 * (1.0 - y ** 2.0)
