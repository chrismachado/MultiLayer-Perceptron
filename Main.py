# for weight initialization check: https://towardsdatascience.com/weight-initialization-techniques-in-neural-networks-26c649eb3b78
import pandas as pd
import numpy as np
import time

from neural_network.MLP import MLP
from utilities.Problems import Problem
from utilities.KFold import KFold

# from sklearn.model_selection import train_test_split
from utilities.Realization import Realization

def main():

    act_func = 'logistic'
    desc_prob = 'dermatology'
    prob = Problem(problem=desc_prob, act_func=act_func)
    X, y = prob.get_dataset()

    import math
    print(math.floor(X.shape[0] * 0.7))
    print(X.shape[0])
    # print(y)

    input_size = X.shape[1]
    hidden_size = None
    output_size = y[0].shape[0]

    mlp = MLP(input_size, hidden_size, output_size,
              hidden_act_func=act_func,
              output_act_func=act_func,
              epoch=50)

    Realization(problem=desc_prob, k=5).execution(X=X,
                                             y=y,
                                             clf=mlp,
                                             num=1)


if __name__ == '__main__':
    main()