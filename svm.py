# -*- encoding: utf-8 -*-
"""
    Author:
    Name:
    Describe:
"""
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import random
import warnings
from sklearn.svm import SVC

def run():
    #
    data_train = pd.read_csv('./data/NSL/train.csv', header=0)
    data_test = pd.read_csv('./data/NSL/test.csv', header=0)
    #
    X_train = data_train[range(41)]
    y_train = data_train.iloc[:, -1]
    X_test = data_test[range(41)]
    y_test = data_test.iloc[:, -1]
    print y_train.value_counts()
    print y_test.value_counts()
    #
    clf = SVC()
    # clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    #
    print confusion_matrix(y_test, y_pred)
    print classification_report(y_test, y_pred)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    run()
