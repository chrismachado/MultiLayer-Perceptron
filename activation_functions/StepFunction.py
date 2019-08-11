import numpy as np


class StepFunction(object):
    '''
    This class is responsible for step functions operations.
    '''

    @staticmethod
    def function(u):
        return u

    @staticmethod
    def derivative(u):
        return 1
