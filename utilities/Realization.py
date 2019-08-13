from utilities.KFold import KFold
from sklearn.model_selection import train_test_split

import numpy as np
import time

class Realization(object):
    def __init__(self, problem):
        self.problem = problem

    def execution(self, X, y, clf, num=20):
        accuracy = list()
        kfold = KFold(k=5, hidden_neurons_list=(2, 4, 6, 8, 10))
        hidden_neuron = None
        timer_list = list()
        weights_w = list()
        weights_m = list()
        best_perform = list()
        hidden_accuracy_list = list()
        hidden_hitrate_list = list()
        generic_result_list = list()

        for _ in range(num):
            clf.shuffle_(X, y)

            print("Execution [%d]" % (_ + 1))
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

            timer = time.time()
            hidden_neuron, hidden_accuracy, hidden_hitrate = kfold.best_result(clf=clf,
                                                                               X_train=X_train,
                                                                               y_train=y_train,
                                                                               iteration=_+1)
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
            hitrate = clf.test(X_test, y_test)
            accuracy.append(hitrate)
            print("\tHit rate %.2f%%" % (hitrate * 100))

            print("-------------------------------------------------------")

        best_index, worst_index = self.evaluate_exec(accuracy=accuracy)

        print("Accuracy      : %4.2f%%" % (np.mean(accuracy) * 100))
        print("Desvio Padrão : %.6f%%" % (np.std(accuracy) * 100))
        #
        with open("log/mlp-p(%s)-f(%s)-i%s-h%s-o%s" % (self.problem,
                                           clf._output_act_func_,
                                           X.shape[1],
                                           best_perform[best_index],
                                           y.shape[1]), 'w') as f:

            f.write("HIDDEN ACTIVATION FUNCTION: %s\n" % clf._hidden_act_func_)
            f.write("OUTPUT ACTIVATION FUNCTION: %s\n\n" % clf._output_act_func_)

            f.write("INPUT SIZE: %02d\n" % clf._input_neurons)
            f.write("HIDDEN SIZE: %02d\n" % best_perform[best_index])
            f.write("OUTPUT SIZE: %02d\n\n" % clf._output_neurons)

            f.write("EPOCHS: %03d\n\n" % clf._epoch)

            f.write("TIME: %3.6fs\n\n" % timer_list[best_index])

            f.write("BEST REALIZATION: %2d\n" % (best_index + 1))
            f.write("WORST REALIZATION: %2d\n\n" % (worst_index + 1))

            f.write("CROSS VALIDATION (BEST)\n")
            f.write("|========================================|\n| ")
            for i in range(len(kfold.hidden_neurons_list)):
                f.write("%02d\t |" % kfold.hidden_neurons_list[i])
            f.write("\n|")
            for i in range(len(kfold.generic_result)):
                f.write("%.2f\t |" % (generic_result_list[best_index][i] * 100))
            f.write("\n|========================================|\n\n")
            for i in range(len(accuracy)):
                f.write("=> Realização %d : Taxa de acerto %.2f%%\n" % ((i + 1), (accuracy[i]) * 100))

            f.write("\nRESULTADO FINAL\n")
            f.write("|========================================|\n")
            f.write("|==  Acurácia : %4.2f%%                   |\n" % (np.mean(accuracy) * 100))
            f.write("|==  Desvio Padrão : %.6f%%           |\n" % np.std(accuracy))
            f.write("|========================================|\n")

    def evaluate_exec(self, accuracy):
        max_acc_value = accuracy[0]
        min_acc_value = accuracy[0]
        imax_ = 0
        imin_ = 0

        for index in range(1, len(accuracy)):
            if max_acc_value <= accuracy[index]:
                imax_ = index
                max_acc_value = accuracy[index]
            if min_acc_value >= accuracy[index]:
                imin_ = index
                min_acc_value = accuracy[index]
        return imax_, imin_