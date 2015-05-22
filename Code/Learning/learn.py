from numpy import squeeze
import cPickle
from sklearn import svm, cross_validation, preprocessing, ensemble
import pandas as pd
from utils import *

log("load data")
data = cPickle.load(open(clean_data_dir + "Truncated_SVD_100_factors.csc"))

train_labels = pd.read_csv(labels_dir + 'orange_large_train_toy.labels').values
train_labels = squeeze(train_labels)

n_train = len(train_labels)
train_labels = train_labels[:n_train]
train_data = data[:n_train, ]
train_data = preprocessing.scale(train_data)

log("cross_val_score")
clf = svm.SVC(kernel='poly', degree=1)
#clf = ensemble.GradientBoostingClassifier(learning_rate=0.5, random_state=0)
scores = cross_validation.cross_val_score(clf, train_data, train_labels, cv=5,
                                          verbose=4, scoring='roc_auc')

log("n_train %i. Accuracy: %0.2f (+/- %0.2f)" % (n_train, scores.mean(), scores.std() * 2))
log("done")