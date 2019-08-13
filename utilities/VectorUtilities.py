import numpy as np


class VectorUtilities(object):

    @staticmethod
    def shuffle_(X, y):
        state = np.random.get_state()
        np.random.shuffle(X)
        np.random.set_state(state)
        np.random.shuffle(y)

    @staticmethod
    def normalize_(X):
        for i in range(X.shape[1]):
            max_ = max(X[:, i])
            min_ = min(X[:, i])
            for j in range(X.shape[0]):
                X[j, i] = (X[j, i] - min_) / (max_ - min_)

    @staticmethod
    def evaluate_exec(accuracy):
        max_acc_value = accuracy[0]
        min_acc_value = accuracy[0]
        imax_ = 0
        imin_ = 0

        for index in range(1, len(accuracy)):
            if max_acc_value <= accuracy[index]:
                imax_ = index
                max_acc_value = accuracy[index]
            if min_acc_value >= accuracy[index]:
                imin_ = index
                min_acc_value = accuracy[index]
        return imax_, imin_