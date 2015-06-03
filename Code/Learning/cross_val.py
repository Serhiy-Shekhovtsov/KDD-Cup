import scipy
import cPickle
import csv
import random
from sparsesvd import sparsesvd
import numpy as np
import matplotlib.pyplot as plt
from numpy import squeeze
import pandas as pd
from sklearn import svm, cross_validation, preprocessing, ensemble
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.learning_curve import learning_curve
from decompose import load_decomposed
from plot_learning_curve import plot_learning_curve
from utils import *

log("quick cross val")

n_factors = 255
max_features = .25  # 1.0
subsample = 1.  # 1.0
learning_rate = .1  # .1

log("load data")
train_data = load_decomposed(n_factors)
train_data = preprocessing.scale(train_data)

train_indices = np.load(clean_data_dir + data_size + "train_indices.npy")
labels = squeeze(pd.read_csv(labels_dir + data_size + 'appetency.labels', header=None).values)
labels[labels == -1] = 0
train_labels = labels[train_indices]

log("fit")
clf = ensemble.GradientBoostingClassifier(random_state=9, subsample=subsample,
                                          max_features=max_features)

scores = cross_validation.cross_val_score(clf, train_data, train_labels, cv=7,
                                          verbose=4, scoring='roc_auc')

log("score = %0.3f, n_factors: %i, subsample: %0.2f, learning_rate: %0.4f, max_features: %0.2f"
    % (scores.mean(), n_factors, subsample, learning_rate, max_features))

log("done")
