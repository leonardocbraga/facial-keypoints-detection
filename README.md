# Project

This task refers to the [Kaggle competition] (https://www.kaggle.com/c/facial-keypoints-detection/) to detect the location of keypoints on face images. It consists of files to fit the training data, load data from training and test set, predict an outcome given an input X and generate the submission file. Also, there is a file to test predictions on some examples of test set.

# Platform

The source code is written in [Python] (https://www.python.org/) with [scikit-learn] (http://scikit-learn.org/) whereas they can afford Machine Learning Algorithms.

# Data
The training and test sets can be obtained at [the challenge data page] (https://www.kaggle.com/c/facial-keypoints-detection/data). The training set is a list of 7049 training images and the test set is a list of 1783 test images.

# Solution

The linear model used to fit the data was Ridge regression of scikit-learn package and part of the solution was provided by following the [Daniel Nouri's tutorial](http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/). There are six classifiers called specialists, and each of them is designed for training specific parts of face: eye centers, eye corners, nose tip, mouth corners, mouth center and eyebrows.

# Score

Getting it done gives a score of 3.35170 on the [Public Leaderboard](https://www.kaggle.com/c/facial-keypoints-detection/leaderboard) taking to the top 50 (until now).
