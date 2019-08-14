# for weight initialization check: https://towardsdatascience.com/weight-initialization-techniques-in-neural-networks-26c649eb3b78
import pandas as pd
import numpy as np
import time

from neural_network.MLP import MLP
from utilities.Problems import Problem
from utilities.KFold import KFold

# from sklearn.model_selection import train_test_split
from utilities.Realization import Realization

from matplotlib import pyplot as plt
from utilities.PlotLib import PlotUtil

def main():

    act_func = 'logistic'
    output_act_func = 'regression'
    desc_prob = 'regression'
    # desc_prob = 'xor'
    # desc_prob = 'column'
    # desc_prob = 'iris'
    # desc_prob = 'dermatology'
    # desc_prob = 'breast_cancer'
    prob = Problem(problem=desc_prob, act_func=act_func)
    X, y = prob.get_dataset()

    input_size = X.shape[1]
    hidden_size = 4
    output_size = y[0].shape[0]

    mlp = MLP(input_size, hidden_size, output_size,
              hidden_act_func=act_func,
              output_act_func=act_func,
              epoch=10)

    # plt.plot(X, y, 'bo', markersize=2)


    mlp.fit(X, y)
    print(mlp.hidden_neurons_layer[0].get_w())
    y_predict_list = mlp.estimate(X=X)
    # print(y_predict_list)

    plt.plot(X, y_predict_list, 'yo')
    plt.show()

    # Realization(problem=desc_prob, k=2).execution(X=X,
    #                                          y=y,
    #                                          clf=mlp,
    #                                          num=1)


if __name__ == '__main__':
    main()