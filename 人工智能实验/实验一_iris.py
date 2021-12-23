# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model


def laod_data():
    iris = datasets.load_iris()
    X_train = iris.data
    y_train = iris.target
    return train_test_split(X_train, y_train,
                            test_size=0.25, random_state=0, stratify=y_train)  # stratify分层


def test_LogisticRegression(*data):
    X_train, X_test, y_train, y_test = data
    regr = linear_model.LogisticRegression(solver='liblinear')
    regr.fit(X_train, y_train)
    print('Coefficients:%s, intercept %s' % (regr.coef_, regr.intercept_))
    print("Score:%.2f" % regr.score(X_test, y_test))



if __name__ == '__main__':
    X_train, X_test, y_train, y_test = laod_data()
    test_LogisticRegression(X_train, X_test, y_train, y_test)


