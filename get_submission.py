import pandas as pd
import numpy as np
import csv as csv
import cPickle

from predict import predict
from load_data import load_data
from load_data import LOOKUP_FILE_NAME

def get_submission():
    with open('model.pkl', 'rb') as fid:
        model = cPickle.load(fid)

    X = load_data(test = True)[0]

    y = predict(X)

    lookup_df = pd.read_csv(LOOKUP_FILE_NAME, header=0)

    points = lookup_df[['ImageId', 'FeatureName']].apply(lambda x: y.loc[x[0] - 1, x[1]], axis = 1)

    ids = range(1, points.shape[0] + 1)

    predictions_file = open("submission.csv", "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["RowId", "Location"])
    open_file_object.writerows(zip(ids, points))
    predictions_file.close()
    print 'Done.'
