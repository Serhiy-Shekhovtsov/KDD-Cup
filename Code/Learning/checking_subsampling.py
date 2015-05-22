import cPickle
import csv
import numpy as np
import random
from numpy import squeeze
import pandas as pd
from sklearn import svm, cross_validation, preprocessing, ensemble
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score
from utils import *

n_factors = 200

operation_name = "accuracy per subsample. gbm. 200 factors"
log(operation_name)

results_file_name = results_dir + operation_name + datetime.now().strftime('%Y-%m-%d %H-%M-%S') + ".csv"
results = []

data = cPickle.load(open(clean_data_dir + "train_val_set.csc"))
train_indices = np.load(clean_data_dir + "train_indices.npy")
train_labels = pd.read_csv(labels_dir + 'orange_large_train_appetency.labels', header=None)
train_labels = squeeze(train_labels.values)[train_indices]
train_labels[train_labels == -1] = 0

log("TruncatedSVD.fit_transform data")
svd = TruncatedSVD(n_components=n_factors, random_state=42)
train_data = svd.fit_transform(data)
n_items = train_data.shape[0]
indices = np.arange(0, n_items)

train_data = preprocessing.scale(train_data)

for subsample in np.arange(.1, 1.1, .1):
    log("calculating train and test accuracy, subsample = %0.1f" % subsample)

    clf = ensemble.GradientBoostingClassifier(random_state=9, subsample=subsample)

    log("running cross_val_score")
    scores = cross_validation.cross_val_score(clf, train_data, train_labels, cv=3,
                                              verbose=4, scoring='roc_auc')

    log("train on whole train val set")
    clf.fit(train_data, train_labels)

    log("predict")
    train_result = clf.predict_proba(train_data)
    train_result = train_result[:, 1]
    train_accuracy = roc_auc_score(train_labels, train_result)
    log("subsample %0.1f, train accuracy: %0.2f, test accuracy: %0.2f" % (subsample, train_accuracy, scores.mean()), bcolors.OKBLUE)

    results.append([subsample, train_accuracy, scores.mean()])

    with open(results_file_name, 'w') as fp:
        a = csv.writer(fp, delimiter=',', lineterminator='\n')
        a.writerows(results)


log("done")
