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
    X, y = Problem(problem='iris', act_func=act_func).dataset()

    input_size = X.shape[1]
    hidden_size = None
    output_size = y[0].shape[0]
    mlp = MLP(input_size, hidden_size, output_size,
              hidden_act_func=act_func,
              output_act_func=act_func,
              epoch=10)
    Realization(problem='iris').execution(X=X, y=y, clf=mlp, num=5)


if __name__ == '__main__':
    main()