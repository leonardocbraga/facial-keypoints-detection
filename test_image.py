import pandas as pd
import numpy as np
import csv as csv
from matplotlib import pyplot as plt
import math
from sklearn import linear_model, decomposition
from sklearn.neural_network import MLPClassifier
from sklearn import datasets, svm, metrics
import cPickle
from predict import predict
from load_data import load_data

with open('model.pkl', 'rb') as fid:
    model = cPickle.load(fid)

X = load_data(test = True)[0]

y = predict(X)
	
for i in [3,4,5]:#xrange(4275, 4280):
    plt.figure()
    plt.imshow(np.reshape(X[i, 0::], (96, 96)), cmap='Greys_r')
    for j in xrange(0, len(y.columns), 2):
        plt.plot(y.iloc[i, j], y.iloc[i, j+1], 'bo')

plt.show()