import cPickle
import csv
import random
from sparsesvd import sparsesvd
import numpy as np
import matplotlib.pyplot as plt
from numpy import squeeze
import pandas as pd
from scipy.sparse import vstack
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

n_factors = 255
max_features = .1  # 1.0
subsample = 1.  # 1.0
learning_rate = .1  # .1
log("quick check started")

alg = "GBM"

train_data = load_decomposed(n_factors)
train_data = preprocessing.scale(train_data)

train_indices = np.load(clean_data_dir + data_size + "train_indices.npy")
train_labels = pd.read_csv(labels_dir + data_size + 'appetency.labels', header=None)

train_labels = squeeze(train_labels.values)[train_indices]
train_labels[train_labels == -1] = 0

#n_true = sum(train_labels == 1)
#train_labels = np.concatenate((train_labels[train_labels == 1], train_labels[train_labels == 0][:n_true]), axis=1)
#train_data = np.concatenate((train_data[train_labels == 1, :],  train_data[train_labels == 0, :][:n_true, :]), axis=0)

clf = None
if alg == "SVM":
    clf = svm.SVC(probability=True, kernel='linear', class_weight='auto')
elif alg == "SGD":
    clf = SGDClassifier(loss='log')
elif alg == "GBM":
    clf = ensemble.GradientBoostingClassifier(max_features=max_features,
                                              subsample=subsample, learning_rate=learning_rate)

log("cross_val_score")
plot_title = data_size + "%s n_factors: %i, subsample: %0.2f, learning_rate: %0.4f, max_features: %0.2f" \
                         % (alg, n_factors, subsample, learning_rate, max_features)

scores = plot_learning_curve(clf, plot_title, train_data, train_labels, cv=3, verbose=4,
                             scoring='roc_auc')

plt.show()

log("done")
