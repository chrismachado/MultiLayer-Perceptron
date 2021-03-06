import pandas as pd
import numpy as np

from utilities.Dataset import Dataset
from utilities.VectorUtilities import VectorUtilities

class Problem(object):

    def __init__(self, problem='iris', act_func='default'):
        self.__problem = problem
        self.__act_func = act_func
        self.dataset = Dataset(act_func=act_func)
        self.vu = VectorUtilities()

    def get_dataset(self):
        print("Collecting dataset from -> ", end='')
        X, y = (None, None)

        if self.__problem == 'iris':
            X, y = self.dataset.iris()

        elif self.__problem == 'column':
            X, y = self.dataset.vertebral_column()

        elif self.__problem == 'breast_cancer':
            X, y = self.dataset.breast_cancer()

        elif self.__problem == 'dermatology':
            X, y = self.dataset.dermatology()

        elif self.__problem == 'xor':
            X, y = self.dataset.xor()

        elif self.__problem == 'regression':
            X, y = self.dataset.regression()
        else:
            print("Invalid dataset selected.")
            exit(0)

        if self.__act_func == 'tanh':
            aux = -1
        else:
            aux = 0

        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                if y[i][j] == 0:
                    y[i][j] = aux

        if self.__problem != 'regression':
            self.vu.normalize_(X=X)

        return X, y