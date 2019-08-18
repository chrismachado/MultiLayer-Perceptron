import numpy as np
import time
from sklearn.metrics import mean_squared_error


class KFold(object):

    def __init__(self, k=5, hidden_neurons_list=(2, 4, 6, 8, 10)):
        self.k = k
        self.hidden_neurons_list = hidden_neurons_list
        self.generic_result = list()

    def k_split(self, X, y):
        X_splitted = np.split(X, self.k)
        y_splitted = np.split(y, self.k)

        return X_splitted, y_splitted

    def k_train(self, clf, X_train, y_train):
        X_splitted_train, y_splitted_train = self.k_split(X_train, y_train)
        accuracy_n_hidden = list()

        for i in range(self.k):
            print("\t\tExecutting Iteration %d..." % (i+1), end=' ')
            X_splitted_train_without_k = self.make_new_list_without_k(_list=X_splitted_train,
                                                                      k=i)
            y_splitted_train_without_k = self.make_new_list_without_k(_list=y_splitted_train,
                                                                      k=i)

            clf.fit(X=X_splitted_train_without_k,
                    y=y_splitted_train_without_k)

            accuracy_n_hidden.append(clf.predict(X=X_splitted_train[i],
                                                  y=y_splitted_train[i]))

            print("Computation %.6f" % accuracy_n_hidden[-1])

        return np.mean(accuracy_n_hidden), accuracy_n_hidden

    def best_result(self, clf, X_train, y_train, iteration):
        timerf = 0

        print("Initializing K-Fold Cross Validation...")
        accuracy_hidden_value = list()
        accuracy_hidden_list = list()

        print("Executing %d-Fold Cross Validation" % self.k)

        for hidden_neurons in self.hidden_neurons_list:
            print("\tHidden neuron size : %d" % hidden_neurons)

            clf._hidden_neurons = hidden_neurons

            timer = time.time()
            ahv, ahl = self.k_train(clf=clf, X_train=X_train, y_train=y_train)
            timerf = time.time() - timer

            accuracy_hidden_value.append(ahv)
            accuracy_hidden_list.append(ahl)

            if clf._output_act_func_ == 'regression':
                print("\t\tMSE : %.5f" % ahv)
                print("\t\tRMSE : %.5f" % np.sqrt(ahv))
            else:
                print("\t\tAccuracy : %.6f" % ahv)

            print("\t\tArray → %s" % ahl)

        best = accuracy_hidden_value.index(max(accuracy_hidden_value))
        self.generic_result = accuracy_hidden_value

        if self.hidden_neurons_list[0] % 2 == 0:
            type_ = "par"
        else:
            type_ = "impar"

        with open("kfold_log/kfcv-e%s-%s" % (iteration, type_), 'w') as f:

            f.write("TIME: %3.2f s\n" % timerf)
            f.write("CROSS VALIDATION \n")
            f.write("BEST HIDDEN SIZE: %2d \n" % self.hidden_neurons_list[best])
            f.write("ACCURACY : %2.2f\n\n" % (np.mean(accuracy_hidden_value)))
            f.write("|========================================|\n| ")
            for i in range(len(self.hidden_neurons_list)):
                f.write("%02d\t |" % self.hidden_neurons_list[i])
            f.write("\n|")
            for i in range(len(self.generic_result)):
                f.write("%.2f\t |" % (self.generic_result[i] * 100))
            f.write("\n|========================================|\n\n")

        return self.hidden_neurons_list[best], accuracy_hidden_value[best], accuracy_hidden_list[best]

    @staticmethod
    def make_new_list_without_k(_list, k):
        nw_list = list()
        for i in range(len(_list)):
            if i != k:
                for j in range(len(_list[i])):
                    nw_list.append(_list[i][j])

        nw_list = np.array(nw_list)

        return nw_list

