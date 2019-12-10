# -*- encoding = utf-8 -*-

import numpy as np
from math import sqrt
from collections import Counter


def accuracy(y_test, y_predict):
    return np.sum(y_predict == y_test) / len(y_test)


class KNNClassifier:

    def __init__(self, k):
        self.k = k
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        self._X_train = X_train
        self._y_train = y_train
        return self

    def _predict(self, x):
        distances = [ sqrt(np.sum((x_train - x)**2)) for x_train in self._X_train]
        nearest = np.argsort(distances)

        topK_y = [self._y_train[i] for i in nearest[: self.k]]
        votes = Counter(topK_y)

        return votes.most_common(1)[0][0]

    def predict(self, X_predict):
        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)

    def score(self, x_test, y_test):
        y_predict = self.predict(x_test)
        return accuracy(x_test, y_predict)

    def __repr__(self):
        return "KNN(k=%d)" % self.k


