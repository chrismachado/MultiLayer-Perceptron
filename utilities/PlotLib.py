import itertools

from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from utilities.VectorUtilities import VectorUtilities

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

        plt.scatter(X[:, 0], X[:, 1])

        print("\nCreating colormap...", end=' ')
        for x1, x2 in Z:
            predict = clf.around(clf.feedfoward_output([x1, x2]))

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
    def plot_decision_boundary(X, y, c, clf, classifier, X_highlights, problem):
        plt.figure()
        plt.title("BOUNDARY DECISION %s " % classifier)
        # Set min and max values and give it some padding
        x_min, x_max = X[:, 0].min() - .2, X[:, 0].max() + .2
        y_min, y_max = X[:, 1].min() - .2, X[:, 1].max() + .2
        h = 0.01
        # Generate a grid of points with distance h between them
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Predict the function value for the whole grid
        Z = clf.predict_1D(np.c_[xx.ravel(), yy.ravel()])[:, 0]
        Z = Z.reshape(xx.shape)

        # Plot the contour and training examples
        plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, levels=1)

        for x_, target in zip(X, y):
            if np.array_equal(target, [0, 1]):
                marker = '^'
            else:
                marker = 'o'
            plt.scatter([x_[0]], x_[1], c='y', edgecolors='k', marker=marker)

        for xx1, xx2 in X_highlights:
            plt.plot(xx1, xx2, 'ko', fillstyle='none', markersize=8)

        plt.savefig("images/%s-%s-(%s)-%s.png" % ('cmp', classifier, problem, clf._output_act_func_), format='png')
        plt.show()

    @staticmethod
    def plot_regression(X, y, clf, problem, act_func, classifier):
        plt.figure()

        argmin, argmax = X.min() - .2, X.max() + .2
        h = 0.01
        print(argmax, argmin)
        xx = np.arange(argmin, argmax, h)
        xx = np.reshape(xx, newshape=(len(xx), 1))
        print(xx.shape)
        # print(clf.predictions(xx))

        plt.scatter(X, y)
        plt.scatter(xx, clf.predictions(xx), c='r', linestyle='-', s=2)
        # plt.scatter(X, clf.predictions(X), c='r', linestyle='-')
        plt.title("Regression → %s (%s,%s,%s)" % (classifier,
                                                  clf._input_neurons,
                                                  clf._hidden_neurons,
                                                  clf._output_neurons))
        plt.savefig("images/curve-%s(%s)-%s-(%s,%s,%s).png" % (classifier, problem, act_func,
                                                               clf._input_neurons,
                                                               clf._hidden_neurons,
                                                               clf._output_neurons), format='png')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

    @staticmethod
    def plot_conf_matrix(classifier, predict, desired_label, chosen_base, n_labels):
        plt.figure()
        plt.rcParams['figure.figsize'] = (11, 7)

        predict_conv = np.array(predict)
        desired_conv = desired_label

        if n_labels >= 2:
            predict_conv = VectorUtilities.convert_labels(np.array(predict), n_labels)
            desired_conv = VectorUtilities.convert_labels(desired_label, n_labels)

        cnf_matrix = confusion_matrix(desired_conv, predict_conv)
        np.set_printoptions(precision=2)

        class_names = ['Class ' + str(i) for i in range(n_labels)]

        plt.figure()
        PlotUtil.plot_confusion_matrix(cnf_matrix, classes=n_labels, model=classifier,
                              chosen_base=chosen_base, title='CONFUSION MATRIX - {}'.format(chosen_base))
        # plt.show()

        return cnf_matrix

    @staticmethod
    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Matriz de Confusão',
                              cmap=plt.cm.Blues,
                              chosen_base='Iris',
                              model='MLP'):

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Matriz de confusão normalizada")
        else:
            print('Confusion Matrix')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        plt.xticks([])
        plt.yticks([])

        fmt = '.3f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig('images/cm-{}-({})-({})'.format(model, chosen_base, classes))
        plt.tight_layout()