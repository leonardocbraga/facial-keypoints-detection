import pandas as pd
import numpy as np
import math
import cPickle

from sklearn.linear_model import RidgeCV
from load_data import load_data

SPECIALIST_SETTINGS = [
    dict(
        name = 'eye_center',
        columns=(
            'left_eye_center_x', 'left_eye_center_y',
            'right_eye_center_x', 'right_eye_center_y',
            ),
        flip_indices=((0, 2), (1, 3)),
        ),

    dict(
        name = 'nose',
        columns=(
            'nose_tip_x', 'nose_tip_y',
            ),
        flip_indices=(),
        ),

    dict(
        name = 'mouth',
        columns=(
            'mouth_left_corner_x', 'mouth_left_corner_y',
            'mouth_right_corner_x', 'mouth_right_corner_y',
            'mouth_center_top_lip_x', 'mouth_center_top_lip_y',
            ),
        flip_indices=((0, 2), (1, 3)),
        ),

    dict(
        name = 'mouth_bottom',
        columns=(
            'mouth_center_bottom_lip_x',
            'mouth_center_bottom_lip_y',
            ),
        flip_indices=(),
        ),

    dict(
        name = 'eye_sides',
        columns=(
            'left_eye_inner_corner_x', 'left_eye_inner_corner_y',
            'right_eye_inner_corner_x', 'right_eye_inner_corner_y',
            'left_eye_outer_corner_x', 'left_eye_outer_corner_y',
            'right_eye_outer_corner_x', 'right_eye_outer_corner_y',
            ),
        flip_indices=((0, 2), (1, 3), (4, 6), (5, 7)),
        ),

    dict(
        name = 'eyebrow',
        columns=(
            'left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y',
            'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y',
            'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y',
            'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y',
            ),
        flip_indices=((0, 2), (1, 3), (4, 6), (5, 7)),
        ),
    ]

def fit():
    model = dict(classifiers = np.empty(0), columns = np.empty(0))

    for setting in SPECIALIST_SETTINGS:
        cols = setting['columns']
        name = setting['name']
        print 'Training ' + name

        X, y = load_data(columns = cols, flip_indices = setting['flip_indices'])

        classifier = RidgeCV()
        classifier.fit(X, y)

        model['classifiers'] = np.append(model['classifiers'], classifier)
        model['columns'] = np.append(model['columns'], cols)

    with open('model.pkl', 'wb') as fid:
        cPickle.dump(model, fid)
