from matplotlib import pyplot as plt
import numpy as np


class PlotUtil(object):
    @staticmethod
    def plot_decision(X, clf, problem, act_func, X_highlights=None):
        xx1_max, xx1_min = X[:, 0].max() + 0.2, X[:, 0].min() - 0.2
        xx2_max, xx2_min = X[:, 1].max() + 0.2, X[:, 1].min() - 0.2

        xx1, xx2 = np.meshgrid(np.arange(xx1_min, xx1_max, 0.035), np.arange(xx2_min, xx2_max, 0.035))
        Z = np.array([xx1.ravel(), xx2.ravel()]).T

        aux = 0 if act_func != 'tanh' else -1
        s = 25
        marker = 's'

        print("\nCreating colormap...", end=' ')
        for x1, x2 in Z:
            predict = clf.around(clf.predict([x1, x2]))

            if np.array_equal(predict, np.array([1, aux])):
                plt.scatter(x1, x2, c='#800D9F', s=s, marker=marker)

            elif np.array_equal(predict, np.array([aux, 1])):
                plt.scatter(x1, x2, c='#0083FF', s=s, marker=marker)

        print("Done.")

        for xx1, xx2 in X_highlights:
            plt.plot(xx1, xx2, 'ko', fillstyle='none', markersize=8)

        # ColorMap Perceptron
        plt.xlabel('x0')
        plt.ylabel('x1')
        plt.title('MLP → ColorMap')

        plt.savefig("images/%s(%s)-%s.png" % ('cmp', problem, act_func), format='png')
        plt.show()