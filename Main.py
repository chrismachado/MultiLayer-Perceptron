# for weight initialization check: https://towardsdatascience.com/weight-initialization-techniques-in-neural-networks-26c649eb3b78
import pandas as pd
import numpy as np
import time
import copy as cp

from neural_network.MLP import MLP
from utilities.Problems import Problem
from utilities.KFold import KFold
from utilities.PlotLib import PlotUtil
from utilities.VectorUtilities import VectorUtilities

from sklearn.model_selection import train_test_split
from utilities.Realization import Realization

from matplotlib import pyplot as plt
from utilities.PlotLib import PlotUtil

#%config InlineBackend.figure_format = 'retina'
plt.style.use('bmh')


def main():

    hidden_act_func = 'logistic'
    # output_act_func = 'logistic'

    output_act_func = 'regression'
    desc_prob = 'regression'

    # desc_prob = 'xor'
    # desc_prob = 'column'
    # desc_prob = 'iris'
    # desc_prob = 'dermatology'
    # desc_prob = 'breast_cancer'
    prob = Problem(problem=desc_prob, act_func=output_act_func)
    X, y = prob.get_dataset()

    input_size = X.shape[1]
    hidden_size = 4
    output_size = y[0].shape[0]

    mlp = MLP(input_size, hidden_size, output_size,
              hidden_act_func=hidden_act_func,
              output_act_func=output_act_func,
              epoch=500)

    # Realization(problem=desc_prob, k=5).execution(X=X,
    #                                          y=y,
    #                                          clf=mlp,
    #                                          num=2)


    Xold, yold = cp.deepcopy(X), cp.deepcopy(y)

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.5, random_state=0)

    mlp.fit(Xtrain, ytrain)

    PlotUtil.plot_regression(Xold, yold, clf=mlp)
    # yguess = mlp.estimate(Xtest)
    # plt.scatter(Xtest, ytest)
    # plt.plot(Xtest, yguess, 'r.')

    # plt.show()

if __name__ == '__main__':
    main()

