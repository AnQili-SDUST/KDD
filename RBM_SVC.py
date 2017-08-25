# -*-encoding: utf-8 -*-
"""

"""
import time
import numpy as np
from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.svm import SVC

# Load Data
t = time.time()
data_train = pd.read_csv('./data/NSL/train.csv', header=0)
data_test = pd.read_csv('./data/NSL/test.csv', header=0)
X_train = data_train.iloc[:, 0:-1]
X_test = data_test.iloc[:, 0:-1]
X_train = (X_train - np.min(X_train, 0)) / (np.max(X_train, 0) + 0.0001)  # 0-1 scaling
X_test = (X_test - np.min(X_test, 0)) / (np.max(X_test, 0) + 0.0001)  # 0-1 scaling
y_train = data_train.iloc[:, -1]
y_test = data_test.iloc[:, -1]

# Model we will use
svm = SVC()
rbm = BernoulliRBM(random_state=0, verbose=True)

classifier = Pipeline(steps=[('rbm', rbm), ('Support Vector Machine', svm)])
rbm.n_iter = 20
# More components tend to give better prediction performance, but larger
# fitting time
rbm.n_components = 100

# Training RBM-Logistic Pipeline
classifier.fit(X_train, y_train)

# Training SVC
svm = SVC()
svm.fit(X_train, y_train)

print("Logistic regression using RBM features:\n%s\n" % (
    metrics.classification_report(y_test, classifier.predict(X_test))))

print("Logistic regression using raw pixel features:\n%s\n" % (
    metrics.classification_report(y_test, svm.predict(X_test))))
print time.time() - t