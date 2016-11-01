import pandas as pd
import numpy as np
from sklearn.utils import shuffle

TRAINING_FILE_NAME = '../data/training.csv'
TEST_FILE_NAME = '../data/test.csv'
LOOKUP_FILE_NAME = '../data/IdLookupTable.csv'

def load_data(test = False, columns = None, flip_indices = None):
    data_frame = pd.read_csv(TRAINING_FILE_NAME if not test else TEST_FILE_NAME, header=0)
    data_frame['Image'] = data_frame['Image'].apply(lambda im: np.fromstring(im, sep = ' '))

    if columns != None:
        data_frame = data_frame[list(columns) + ['Image']]
    data_frame = data_frame.dropna()

    X = np.vstack(data_frame['Image'].values) / 255.
    X = X.astype(np.float32)

    y = data_frame[data_frame.columns[:-1]].values

    if not test and flip_indices != None:
        indices = np.random.choice(X.shape[0], X.shape[0] / 2, replace=False)

        X = np.reshape(X, (X.shape[0], 96, 96))
        X[indices] = X[indices, ::, ::-1]
        X = np.reshape(X, (X.shape[0], 9216))

        y[indices, ::2] = y[indices, ::2] + 2*(48 - y[indices, ::2])

        for a, b in flip_indices:
            y[indices, a], y[indices, b] = (y[indices, b], y[indices, a])

    if not test:
        X, y = shuffle(X, y, random_state = 0)

    return X, y
