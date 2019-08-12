import numpy as np


class LogisticFunction(object):
    '''
    This class is responsible for logistic functions operations.
    '''

    @staticmethod
    def function(u):
        return 1.0 / (1.0 + np.exp(-u))

    @staticmethod
    def derivative(u):
        y = LogisticFunction.function(u)
        return y * (1.0 - y)
