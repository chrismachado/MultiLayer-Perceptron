
import pandas as pd
import numpy as np
import requests, zipfile, io
import random

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

    #TODO fazer bases xor
    def vertebral_column(self):
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00212/vertebral_column_data.zip'
        print(url)

        r = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        df = pd.read_csv(z.open('column_3C.dat'), header=None, sep=' ')

        y = df.iloc[:, 6].values
        y_ = np.zeros((y.shape[0], 3))

        for i in range(y.shape[0]):
            if y[i] == 'DH':
                y_[i][0] = 1
            elif y[i] == 'SL':
                y_[i][1] = 1
            elif y[i] == 'NO':
                y_[i][2] = 1
        X = df.iloc[:, [i for i in range(5)]].values
        X = np.float_(X)

        return X, y_

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
        X = np.float_(X)

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
        X = np.float_(X)

        return X, y_

    def xor(self):
        print("myself, generating data...")
        base_elements = [[0, 0], [1, 1], [0, 1], [1, 0]]

        X = np.zeros(shape=(200, 2))
        y = np.zeros(shape=(200, 2))

        noise = 0.1

        for i in range(200):
            if i < 50 :
                X[i][0] = 0 + np.random.uniform(-noise, noise)
                X[i][1] = 0 + np.random.uniform(-noise, noise)
                y[i][0] = 1

            if 50 <= i < 100:
                X[i][0] = 0 + np.random.uniform(-noise, noise)
                X[i][1] = 1 + np.random.uniform(-noise, noise)
                y[i][1] = 1

            if 100 <= i < 150:
                X[i][0] = 1 + np.random.uniform(-noise, noise)
                X[i][1] = 0 + np.random.uniform(-noise, noise)
                y[i][1] = 1

            if i >= 150:
                X[i][0] = 1 + np.random.uniform(-noise, noise)
                X[i][1] = 1 + np.random.uniform(-noise, noise)
                y[i][0] = 1

        return X, y

    def regression(self):
        print("myself, generating data...")

        N = 500
        noise = 0.25

        X = np.random.normal(size=N) + np.random.randn(N)
        X = X.reshape((N, 1))

        randoms = np.random.uniform(-noise, noise, N)
        noises = np.reshape(randoms, newshape=(N, 1))
        y = 3.0 * np.sin(X) + 1.0 + noises
        y = 2.0 * np.sin(X) + 3.0 + noises
        y = y.reshape((N, 1))
        return X, y


