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

log("final check started")

n_factors = 470
max_features = .16  # 1.0
subsample = 1.  # 1.0
learning_rate = .1  # .1

log("load data")
full_data = cPickle.load(open(clean_data_dir + data_size + "full_data.csc"))
train_data = cPickle.load(open(clean_data_dir + data_size + "train_val_set.csc"))
test_data = cPickle.load(open(clean_data_dir + data_size + "test_set.csc"))

train_indices = np.load(clean_data_dir + data_size + "train_indices.npy")
test_indices = np.load(clean_data_dir + data_size + "test_indices.npy")
labels = squeeze(pd.read_csv(labels_dir + data_size + 'appetency.labels', header=None).values)

labels[labels == -1] = 0
train_labels = labels[train_indices]
test_labels = labels[test_indices]

# SVD
log("decomposing")
svd = TruncatedSVD(n_components=n_factors, random_state=42)
svd.fit(full_data)

train_data = svd.transform(train_data)
test_data = svd.transform(test_data)

# scaling
log("scaling")
scaler = preprocessing.StandardScaler()
scaler.fit(train_data)
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

log("fit")
clf = ensemble.GradientBoostingClassifier(max_features=max_features)
clf.fit(train_data, train_labels)

test_predictions = clf.predict_proba(test_data)
score = roc_auc_score(test_labels, test_predictions[:, 1])

log("final score = %0.3f" % score)

log("done")
