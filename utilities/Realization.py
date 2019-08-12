from sklearn.model_selection import train_test_split
import numpy as np

class Realization(object):

    @staticmethod
    def execution(X, y, clf, num=20):
        accuray = list()

        for _ in range(num):
            print("Execution [%d]" % (_ + 1))
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

            print("\tFitting...")
            clf.fit(X_train, y_train)
            print("\tFinish fitting.")

            print("\tStarting test...")
            hitrate = clf.test(X_test, y_test)
            accuray.append(hitrate)
            print("\tHit rate %.2f%%" % (hitrate * 100))

            print("-------------------------------------------------------")
        print("Accuracy : %.2f%%" % (np.mean(accuray) * 100))