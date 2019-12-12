# -*- encoding = utf-8 -*-
import numpy as np
from math import sqrt


def accuracy(y_test, y_predict):
    return np.sum(y_predict == y_test) / len(y_test)


def mean_squared_error(y_test, y_predict):
    return np.sum((y_test - y_predict) ** 2) / len(y_test)


def root_mean_squared_error(y_test, y_predict):
    return sqrt(mean_squared_error(y_test, y_predict))


def mean_absolute_error(y_test, y_predict):
    return np.sum(np.absolute(y_test - y_predict)) / len(y_test)


def r2_score(y_test, y_predict):
    return 1 - mean_squared_error(y_test, y_predict) / np.var(y_test)