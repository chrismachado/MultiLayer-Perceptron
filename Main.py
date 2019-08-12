# for weight initialization check: https://towardsdatascience.com/weight-initialization-techniques-in-neural-networks-26c649eb3b78
import pandas as pd
import numpy as np
import time

from neural_network.MLP import MLP
from utilities.Problems import Problem


# from sklearn.model_selection import train_test_split
from utilities.Realization import Realization

def main():

    act_func = 'logistic'
    X, y = Problem(problem='iris', act_func=act_func).dataset()
    input_size = X.shape[1]
    hidden_size = 11
    output_size = y[0].shape[0]
    mlp = MLP(input_size, hidden_size, output_size,
              hidden_act_func=act_func,
              output_act_func=act_func,
              epoch=200)

    Realization().execution(X=X, y=y, clf=mlp, num=20)


    # print("Fitting...", end='\t')
    # start = time.time()
    # mlp.fit(X=X_train, y=y_train)
    # end = (time.time() - start)
    # print("Fitting time : %.6fs" % end)
    #
    # print("Starting test...", end='\t')
    # hit = mlp.test(X=X_test, y=y_test)
    # print("Result from test %.2f%%" % (hit * 100))

    # print("[WEIGHTS] Input → Hidden")
    # for hn in mlp.hidden_neurons_layer:
    #     print(hn._w)
    #
    # print("[WEIGHTS] Hidden → Output")
    # for i_n in mlp.output_neurons_layer:
    #     print(i_n._m)

if __name__ == '__main__':
    main()