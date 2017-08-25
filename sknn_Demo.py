from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sknn.mlp import Classifier, Layer
from sklearn.metrics import accuracy_score, classification_report

iris = load_iris()
X = np.asarray(iris.data, 'float32')
y = np.asarray(iris.target, 'float32')
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=0)

nn = Classifier(
    layers=[
        Layer("Rectifier", units=100),
        Layer("Softmax")],
    learning_rate=0.02,
    n_iter=20, verbose=True)
nn.fit(X_train, y_train)
y_pred = nn.predict(X_test)
print accuracy_score(y_test, y_pred)
print nn.score(X_test, y_test)