import pandas as pd
import numpy as np

class Dataset(object):
    def __init__(self, act_func='default'):
        self.__act_func = act_func

    def iris(self):
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

        return X, y_

    #TODO fazer bases coluna, cancer, dermatology e xor
    def vertebral_column(self):
        return

    def breast_cancer(self):
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'
        print(url)
        df = pd.read_csv(url, header=None)
        y = df.iloc[:, 10].values
        y_ = np.zeros((y.shape[0], 2))

        for i in range(y.shape[0]):
            if y[i] == 2:
                y_[i] = [1, 0]
            elif y[i] == 4:
                y_[i] = [0, 1]

        X = df.iloc[:, [i for i in range(1, 10)]].values

        X = np.where(X == '?', 0, X)
        X = np.int_(X)

        return X, y_

    def dermatology(self):
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/dermatology/dermatology.data'
        print(url)
        df = pd.read_csv(url, header=None)
        y = df.iloc[:, 34].values
        y_ = np.zeros((y.shape[0], 6))

        if self.__act_func == 'tanh':
            aux = -1
        else:
            aux = 0

        for i in range(y.shape[0]):
            y_[i][y[i] - 1] = 1

        X = df.iloc[:, [i for i in range(34)]].values

        X = np.where(X == '?', 0, X)
        X = np.int_(X)

        return X, y_

    def xor(self):
        return
