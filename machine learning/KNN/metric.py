# -*- encoding = utf-8 -*-
import numpy as np


def accuracy(y_test, y_predict):
    return np.sum(y_predict == y_test) / len(y_test)