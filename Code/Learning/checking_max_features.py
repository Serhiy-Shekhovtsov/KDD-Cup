import cPickle
import csv
import numpy as np
import random
from numpy import squeeze
import pandas as pd
from sklearn import svm, cross_validation, preprocessing, ensemble
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score
from decompose import load_decomposed
from utils import *

n_factors = 255

operation_name = data_size + "score per max_features. gbm. %i factors" % n_factors
log(operation_name)

results_file_name = results_dir + operation_name + datetime.now().strftime('%Y-%m-%d %H-%M-%S') + ".csv"

train_data = load_decomposed(n_factors)  # , alg="sparsesvd"
train_data = preprocessing.scale(train_data)
n_items = train_data.shape[0]

train_indices = np.load(clean_data_dir + data_size + "train_indices.npy")
train_labels = pd.read_csv(labels_dir + data_size + 'appetency.labels', header=None)
train_labels = squeeze(train_labels.values)[train_indices]
train_labels[train_labels == -1] = 0

results = []

for max_features in np.arange(.01, 1.1, .01):
    log("calculating train and test accuracy, max_features = %0.2f" % max_features)

    clf = ensemble.GradientBoostingClassifier(random_state=9,  # subsample=subsample,
                                              max_features=max_features)

    log("running cross_val_score")
    scores = cross_validation.cross_val_score(clf, train_data, train_labels, cv=7,
                                              verbose=4, scoring='roc_auc')

    log("train on whole train val set")
    clf.fit(train_data, train_labels)

    log("predict")
    train_result = clf.predict_proba(train_data)
    train_result = train_result[:, 1]
    train_accuracy = roc_auc_score(train_labels, train_result)
    log("max_features %0.2f, train accuracy: %0.4f, test accuracy: %0.4f" % (max_features, train_accuracy, scores.mean()), bcolors.OKBLUE)

    results.append([max_features, train_accuracy, scores.mean()])

    with open(results_file_name, 'w') as fp:
        a = csv.writer(fp, delimiter=',', lineterminator='\n')
        a.writerows(results)


log("done")
