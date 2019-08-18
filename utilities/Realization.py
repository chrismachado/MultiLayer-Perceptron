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
    def __init__(self, classifier,  problem, k=5, hidden_size=(2, 4, 6, 8, 10)):
        self.classifier = classifier
        self.problem = problem
        self.vu = VectorUtilities()
        self.k = k
        self.hidden_size = hidden_size

    def execution(self, X, y, clf, num=20):
        balanced_split = self.vu.balance(problem=self.problem, k=self.k, start=0.2, size=X.shape[0])

        accuracy = list()

        hidden_type = self.hidden_size

        kfold = KFold(k=self.k, hidden_neurons_list=hidden_type)

        timer_list = list()
        weights_w = list()
        weights_m = list()
        best_perform = list()
        hidden_accuracy_list = list()
        hidden_computation_list = list()
        generic_result_list = list()

        X_plot = cp.deepcopy(X)
        y_plot = cp.deepcopy(y)

        for _ in range(num):
            self.vu.shuffle_(X, y)

            print("Execution [%d]" % (_ + 1))

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=balanced_split)

            timer = time.time()
            hidden_neuron, hidden_accuracy, hidden_computation = kfold.best_result(clf=clf,
                                                                               X_train=X_train,
                                                                               y_train=y_train,
                                                                               iteration=_ + 1)
            timer_list.append(time.time() - timer)

            best_perform.append(hidden_neuron)
            hidden_accuracy_list.append(hidden_accuracy)
            hidden_computation_list.append(hidden_computation)
            generic_result_list.append(kfold.generic_result)

            clf._hidden_neuron = hidden_neuron

            print("\tFitting...")
            clf.fit(X_train, y_train)
            weights_w.append(clf.hidden_neurons_layer)
            weights_m.append(clf.output_neurons_layer)
            print("\tFinish fitting.")

            print("\tStarting test...")
            computation = clf.predict(X_test, y_test)
            accuracy.append(computation)
            print("\tComputation %.5f" % computation)

            print("-------------------------------------------------------")

        best_index, worst_index = self.vu.evaluate_exec(accuracy=accuracy)

        clf.init_neurons()
        clf.hidden_neurons_layer = weights_w[best_index]
        clf.output_neurons_layer = weights_m[best_index]

        self.save_weights(clf.hidden_neurons_layer, clf.output_neurons_layer, clf._output_act_func_)

        if clf._output_act_func_ == 'regression':
            print("MSE                  : %4.6f" % np.mean(accuracy))
            print("RMSE                  : %4.6f" % np.mean(np.sqrt(accuracy)))
            PlotUtil.plot_regression(X_test, y_test, clf, problem=self.problem, act_func=clf._output_act_func_, classifier=self.classifier)
        else:
            print("Accuracy            : %4.2f" % np.mean(accuracy))
            # if 1 <= clf._input_neurons <= 2:
            #     PlotUtil.plot_decision_boundary(X_test, clf.boundary_decision(X_test), clf)

        print("Standard Deviations : %.6f" % (np.std(accuracy)))

        # Write in file
        with open("log/%s-p(%s)-f(%s)-i%s-h%s-o%s" % (self.classifier,
                                                           self.problem,
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

            f.write("EPOCHS: %s\n\n" % clf._epoch)

            f.write("TIME: %5.2fm\n\n" % (sum(timer_list) / 60))

            f.write("BEST REALIZATION: %2d\n" % (best_index + 1))
            f.write("WORST REALIZATION: %2d\n\n" % (worst_index + 1))

            f.write("CROSS VALIDATION (BEST)\n")
            f.write("|========================================|\n| ")
            for i in range(len(kfold.hidden_neurons_list)):
                f.write("%02d\t |" % kfold.hidden_neurons_list[i])
            f.write("\n|")
            for i in range(len(kfold.generic_result)):
                f.write("%4.6f\t |" % (generic_result_list[best_index][i] * 100))
            f.write("\n|========================================|\n\n")

            if self.problem == 'regression':
                for i in range(len(accuracy)):
                    f.write("=> Realização %d : MSE %.5f | RMSE %.5f  \n" % ((i + 1), accuracy[i], np.sqrt(accuracy[i])))
            else:
                for i in range(len(accuracy)):
                    f.write("=> Realização %d : Computation %.2f  \n" % ((i + 1), (accuracy[i]) * 100))

            f.write("\nRESULTADO FINAL\n")
            f.write("|========================================|\n")
            if self.problem == 'regression':
                f.write("|==  MSE: %4.6f                     |\n" % (np.mean(accuracy)))
                mrmse = np.sum(np.sqrt(np.array(accuracy)))
                f.write("|==  RMSE: %4.6f                     |\n" % mrmse)
                f.write("|==  Standard Deviation : %.6f             |\n" % np.std(accuracy))
            else:
                f.write("|==  Acurácia : %4.2f                    |\n" % (np.mean(accuracy) * 100))
                f.write("|==  Standard Deviation : %.6f            |\n" % np.std(accuracy))
            f.write("|========================================|\n")



    def save_weights(self, hidden_layer, output_layer, output_act_func):
        hidden_neurons = len(hidden_layer)
        output_neurons = len(output_layer)
        with open("weight_log/%s-p(%s)-f(%s)-h%s-o%s" % (self.classifier,
                                                      self.problem,
                                                      output_act_func,
                                                      hidden_neurons,
                                                      output_neurons), 'w') as f:
            f.write("W: \n")
            for i in range(hidden_neurons):
                f.write("%s\n" % hidden_layer[i].get_w())


            f.write("\nM: \n")
            for j in range(output_neurons):
                f.write("%s\n" % output_layer[j].get_m())
                f.write("\n")
