from utilities.KFold import KFold
from utilities.VectorUtilities import VectorUtilities
from utilities.PlotLib import PlotUtil

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

import numpy as np
import time
import math
import copy as cp


class Realization(object):
    def __init__(self, problem, k=5):
        self.problem = problem
        self.vu = VectorUtilities()
        self.k = k

    def execution(self, X, y, clf, num=20):
        balanced_split = self.balance(k=self.k, start=0.2, size=X.shape[0])
        accuracy = list()

        # hidden_type = (9, 10, 11, 12, 13)
        # hidden_type = (2, 4, 6, 8, 10)
        hidden_type = (4, 6, 8)
        # hidden_type = (3, 5, 7, 9, 11)

        kfold = KFold(k=self.k, hidden_neurons_list=hidden_type)

        timer_list = list()
        weights_w = list()
        weights_m = list()
        best_perform = list()
        hidden_accuracy_list = list()
        hidden_hitrate_list = list()
        generic_result_list = list()

        X_plot = cp.deepcopy(X)
        y_plot = cp.deepcopy(y)

        for _ in range(num):
            self.vu.shuffle_(X, y)

            print("Execution [%d]" % (_ + 1))
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=balanced_split, stratify=y)

            timer = time.time()
            hidden_neuron, hidden_accuracy, hidden_hitrate = kfold.best_result(clf=clf,
                                                                               X_train=X_train,
                                                                               y_train=y_train,
                                                                               iteration=_ + 1)
            timer_list.append(time.time() - timer)

            best_perform.append(hidden_neuron)
            hidden_accuracy_list.append(hidden_accuracy)
            hidden_hitrate_list.append(hidden_hitrate)
            generic_result_list.append(kfold.generic_result)

            clf._hidden_neuron = hidden_neuron

            print("\tFitting...")
            clf.fit(X_train, y_train)
            weights_w.append(clf.hidden_neurons_layer)
            weights_m.append(clf.output_neurons_layer)
            print("\tFinish fitting.")

            print("\tStarting test...")
            hitrate = clf.classify(X_test, y_test)
            accuracy.append(hitrate)
            print("\tHit rate %.2f%%" % (hitrate * 100))

            print("-------------------------------------------------------")

        best_index, worst_index = self.vu.evaluate_exec(accuracy=accuracy)

        clf.init_neurons()
        clf.hidden_neurons_layer = weights_w[best_index]
        clf.output_neurons_layer = weights_m[best_index]

        if X.shape[1] == 2 and self.problem == 'xor':
            plt.plot(X_plot[:50, 0], X_plot[:50, 1], 'bo', mec='k', markersize=5)
            plt.plot(X_plot[50:100, 0], X_plot[50:100, 1], 'y^', mec='k', markersize=5)
            plt.plot(X_plot[100:150, 0], X_plot[100:150, 1], 'y^', mec='k', markersize=5)
            plt.plot(X_plot[150:, 0], X_plot[150:, 1], 'bo', mec='k', markersize=5)

            PlotUtil().plot_decision(X=X_plot, clf=clf, problem=self.problem, act_func=clf._output_act_func_, X_highlights=X_test)

        print("Accuracy      : %4.2f%%" % float((np.mean(accuracy) * 100)))
        print("Desvio Padrão : %.6f%%" % float((np.std(accuracy) * 100)))

        # Write in file

        with open("log/mlp-p(%s)-f(%s)-i%s-h%s-o%s" % (self.problem,
                                                       clf._output_act_func_,
                                                       X.shape[1],
                                                       best_perform[best_index],
                                                       y.shape[1]), 'w') as f:

            f.write("DATASET USED: %s\n" % self.problem)

            f.write("HIDDEN ACTIVATION FUNCTION: %s\n" % clf._hidden_act_func_)
            f.write("OUTPUT ACTIVATION FUNCTION: %s\n\n" % clf._output_act_func_)

            f.write("INPUT SIZE: %02d\n" % clf._input_neurons)
            f.write("HIDDEN SIZE: %02d\n" % best_perform[best_index])
            f.write("OUTPUT SIZE: %02d\n\n" % clf._output_neurons)

            f.write("EPOCHS: %03d\n\n" % clf._epoch)

            f.write("TIME: %5.2fm\n\n" % (sum(timer_list) / 60))

            f.write("BEST plotIZATION: %2d\n" % (best_index + 1))
            f.write("WORST plotIZATION: %2d\n\n" % (worst_index + 1))

            f.write("CROSS VALIDATION (BEST)\n")
            f.write("|========================================|\n| ")
            for i in range(len(kfold.hidden_neurons_list)):
                f.write("%02d\t |" % kfold.hidden_neurons_list[i])
            f.write("\n|")
            for i in range(len(kfold.generic_result)):
                f.write("%.2f\t |" % (generic_result_list[best_index][i] * 100))
            f.write("\n|========================================|\n\n")
            for i in range(len(accuracy)):
                f.write("=> plotização %d : Taxa de acerto %.2f%%\n" % ((i + 1), (accuracy[i]) * 100))

            f.write("\nRESULTADO FINAL\n")
            f.write("|========================================|\n")
            f.write("|==  Acurácia : %4.2f%%                   |\n" % (np.mean(accuracy) * 100))
            f.write("|==  Desvio Padrão : %.6f%%           |\n" % np.std(accuracy))
            f.write("|========================================|\n")

    def balance(self, size, k=5, start=0.2):
        print("Balancing [%s] test size before executions..." % self.problem)
        balanced_division = start
        while (1):
            if math.floor(size * (1 - balanced_division)) % k == 0:
                break
            balanced_division += 0.01

        print("N Samples %d\t|\tEquivalent %2.2f%%" % (math.floor(size * (1 - balanced_division)),
                                                       (100 - 100 * balanced_division)))
        return balanced_division
