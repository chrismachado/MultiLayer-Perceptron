# for weight initialization check: https://towardsdatascience.com/weight-initialization-techniques-in-neural-networks-26c649eb3b78
from neural_network.ELM import ELM
from utilities.Problems import Problem

from utilities.Realization import Realization

def main():

    hidden_act_func = 'logistic'
    output_act_func = 'regression'
    desc_prob = 'regression'

    prob = Problem(problem=desc_prob, act_func=output_act_func)
    X, y = prob.get_dataset()

    input_size = X.shape[1]
    hidden_size = [i for i in range(10, 32, 2)]
    output_size = y[0].shape[0]

    elm = ELM(input_size, None, output_size,
              hidden_act_func=hidden_act_func,
              output_act_func=output_act_func)

    Realization(classifier='elm', problem=desc_prob, k=5, hidden_size=hidden_size).execution(X=X,
                                                  y=y,
                                                  clf=elm,
                                                  num=20)


if __name__ == '__main__':
    main()

