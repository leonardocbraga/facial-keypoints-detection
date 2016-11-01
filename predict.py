import pandas as pd
import numpy as np
import csv as csv
from matplotlib import pyplot as plt
import math
from sklearn import linear_model, decomposition
from sklearn.neural_network import MLPClassifier
from sklearn import datasets, svm, metrics
import cPickle

def predict(X):

    with open('model.pkl', 'rb') as fid:
        model = cPickle.load(fid)

    y = np.empty((X.shape[0], 0))
    for classifier in model['classifiers']:
        y_model = classifier.predict(X)
        y = np.hstack([y, y_model])

    return pd.DataFrame(np.clip(y, 0, 96), columns = model['columns'])
