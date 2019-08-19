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


def main():

    hidden_act_func = 'tanh'
    output_act_func = 'tanh'
    desc_prob = ['xor', 'iris', 'column', 'breast_cancer', 'dermatology']
    p = desc_prob[0]

    for p in desc_prob:
        prob = Problem(problem=p,
                       act_func=output_act_func)
        X, y = prob.get_dataset()

        input_size = X.shape[1]
        hidden_size = (9, 10, 11, 12, 13)
        output_size = y[0].shape[0]

        mlp = MLP(input_size, None, output_size,
                  hidden_act_func=hidden_act_func,
                  output_act_func=output_act_func,
                  epoch=50)

        Realization(classifier='mlp',
                    problem=p, k=5,
                    hidden_size=hidden_size).execution(X=X,
                                             y=y,
                                             clf=mlp,
                                             num=20)




if __name__ == '__main__':
    main()

