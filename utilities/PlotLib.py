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
            predict = clf.around(clf.feedfoward([x1, x2])[2])

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

    @staticmethod
    def plot_decision_boundary(X, y, clf):
        # Set min and max values and give it some padding
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        h = 0.01

        # Generate a grid of points with distance h between them
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Predict the function value for the whole grid
        Z = clf.predict_1D(np.c_[xx.ravel(), yy.ravel()])
        print(Z)
        Z0 = Z[:, 0].reshape(xx.shape)
        Z1 = Z[:, 1].reshape(xx.shape)

        # Plot the contour and training examples
        plt.contourf(xx, yy, Z0, cmap=plt.cm.Spectral, levels=2)
        plt.contourf(xx, yy, Z1, cmap=plt.cm.Spectral, levels=2)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
        plt.show()

    @staticmethod
    def plot_regression(X, y, clf, problem, act_func, classifier):
        plt.scatter(X, y)
        plt.scatter(X, clf.predictions(X), c='r', linestyle='-')
        plt.title("Regression → %s (%s,%s,%s)" % (classifier, clf._input_neurons,  clf._hidden_neurons,  clf._output_neurons))
        plt.savefig("images/curve-%s(%s)-%s-(%s,%s,%s).png" % (classifier, problem, act_func, clf._input_neurons,  clf._hidden_neurons,  clf._output_neurons), format='png')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()