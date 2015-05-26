import cPickle
import csv
from sparsesvd import sparsesvd
import numpy as np
from numpy import squeeze
import pandas as pd
from scipy.sparse import vstack, csc_matrix
from sklearn import svm, cross_validation, preprocessing, ensemble
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.utils.extmath import safe_sparse_dot
from decompose import load_decomposed
from utils import *

operation_name = "find best n_factors(sparsesvd). max_features=.5, subsample=.9 "
log(operation_name)

train_indices = np.load(clean_data_dir + "train_indices.npy")
train_labels = pd.read_csv(labels_dir + 'orange_large_train_appetency.labels', header=None)
train_labels = squeeze(train_labels.values)[train_indices]
train_labels[train_labels == -1] = 0

results_file_name = results_dir + operation_name + datetime.now().strftime('%Y-%m-%d %H-%M-%S') + ".csv"
results = []

for n_factors in range(5, 300, 5):
    log("*** n_factors = %i ***" % n_factors)

    train_data = load_decomposed(n_factors, alg="sparsesvd")
    train_data = preprocessing.scale(train_data)

    clf = SGDClassifier(loss='log', alpha=1000, class_weight={0: 600, 1: 1})
    # clf = ensemble.GradientBoostingClassifier(random_state=0, max_features=.5, subsample=.9)

    log("cross_val_score")
    scores = cross_validation.cross_val_score(clf, train_data, train_labels, cv=3,
                                              verbose=4, scoring='roc_auc')

    log("train on whole train val set")
    clf.fit(train_data, train_labels)

    log("predict")
    train_result = clf.predict_proba(train_data)
    train_result = train_result[:, 1]
    train_accuracy = roc_auc_score(train_labels, train_result)

    log("n_factors %i, train accuracy: %0.4f, test accuracy: %0.4f"
        % (n_factors, train_accuracy, scores.mean()), bcolors.OKBLUE)

    results.append([n_factors, train_accuracy, scores.mean()])

    with open(results_file_name, 'w') as fp:
        a = csv.writer(fp, delimiter=',', lineterminator='\n')
        a.writerows(results)


log("done")
