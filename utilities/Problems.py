import pandas as pd
import numpy as np


class Problem(object):

    def __init__(self, problem='iris', act_func='default'):
        self.__problem = problem
        self.__act_func = act_func

    def dataset(self):
        print("Collecting dataset from → ", end='')
        if self.__problem == 'iris':
            url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
            print(url)
            df = pd.read_csv(url, header=None)
            y = df.iloc[:, 4].values
            y_ = np.zeros((y.shape[0], 3))
            if self.__act_func == 'tanh':
                aux = -1
            else:
                aux = 0

            for i in range(y.shape[0]):
                if y[i] == 'Iris-setosa':
                    y_[i] = [1, aux, aux]
                elif y[i] == 'Iris-versicolor':
                    y_[i] = [aux, 1, aux]
                elif y[i] == 'Iris-virginica':
                    y_[i] = [aux, aux, 1]
            X = df.iloc[:, [0, 1, 2, 3]].values

        self.normalize_(X)

        return X, y_

    @staticmethod
    def normalize_(X):
        for i in range(X.shape[1]):
            max_ = max(X[:, i])
            min_ = min(X[:, i])
            for j in range(X.shape[0]):
                X[j, i] = (X[j, i] - min_) / (max_ - min_)