import pandas as pd
import numpy as np
import time

from neural_network.MLP import MLP
from utilities.Problems import Problem
# for weight initialization check: https://towardsdatascience.com/weight-initialization-techniques-in-neural-networks-26c649eb3b78

def main():

    act_func = 'logistic'
    X, y = Problem(problem='iris', act_func=act_func).dataset()
    input_size = X.shape[1]
    hidden_size = 5
    output_size = y[0].shape[0]
    mlp = MLP(input_size, hidden_size, output_size,
              hidden_act_func=act_func,
              output_act_func=act_func,
              epoch=150)

    print("Fitting...")
    start = time.time()
    mlp.fit(X, y)
    end = (time.time() - start)

    print("[WEIGHTS] Input → Hidden")
    for hn in mlp.hidden_neurons_layer:
        print(hn._w)

    print("[WEIGHTS] Hidden → Output")
    for i_n in mlp.output_neurons_layer:
        print(i_n._m)

    print("Fitting time : %s" % end)


if __name__ == '__main__':
    main()