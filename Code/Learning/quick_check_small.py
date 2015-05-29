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

log("quick check started")

train_data = cPickle.load(open(clean_data_dir + "small_data_full.csc"))
#train_data = preprocessing.scale(train_data)

#train_indices = np.load(clean_data_dir + "train_indices.npy")
train_labels = pd.read_csv(labels_dir + 'orange_small_train_appetency.labels', header=None)

train_labels = squeeze(train_labels.values)
train_labels[train_labels == -1] = 0

# clf = svm.SVC(probability=True)
# clf = GaussianNB()
clf = SGDClassifier(loss='log')
# clf = ensemble.GradientBoostingClassifier(random_state=0, max_features=max_features,
#                                           subsample=subsample, learning_rate=learning_rate)
# clf = ensemble.RandomForestClassifier(random_state=0, max_features=max_features, max_depth=4)

log("cross_val_score")
plot_title = "SGD small data set"
scores = plot_learning_curve(clf, plot_title, train_data, train_labels, cv=3, verbose=4,
                             scoring='roc_auc')

plt.show()

log("done")
