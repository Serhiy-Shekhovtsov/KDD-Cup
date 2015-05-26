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

n_factors = 56
subsample = .6
#max_features = .5
learning_rate = .08

operation_name = "accuracy per N. GBM. 56 factors. subsample 0.9 "
log(operation_name)

results_file_name = results_dir + operation_name + datetime.now().strftime('%Y-%m-%d %H-%M-%S') + ".csv"
results = []

train_indices = np.load(clean_data_dir + "train_indices.npy")
train_labels = pd.read_csv(labels_dir + 'orange_large_train_toy.labels', header=None)
train_labels = squeeze(train_labels.values)[train_indices]
train_labels[train_labels == -1] = 0

train_data = load_decomposed(n_factors, alg="sparsesvd")
train_data = preprocessing.scale(train_data)

n_items = train_data.shape[0]
indices = np.arange(0, n_items)

random.seed(333)
random.shuffle(indices)

for n_examples in range(10, 101, 10):
    log("calculating train and test accuracy, n_features = %i" % n_examples)

    split = int(n_items * n_examples / 100)

    train_indices = indices[0:split]
    temp_data = train_data[train_indices, :]
    temp_labels = train_labels[train_indices]

    clf = ensemble.GradientBoostingClassifier(random_state=9,
                                              subsample=subsample,  # max_features=max_features,
                                              learning_rate=learning_rate)
    # clf = svm.SVC(probability=True)

    log("running cross_val_score")
    scores = cross_validation.cross_val_score(clf, temp_data, temp_labels, cv=3,
                                              verbose=4, scoring='roc_auc')

    log("train on whole train val set")
    clf.fit(temp_data, temp_labels)

    log("predict")
    train_result = clf.predict_proba(temp_data)
    train_result = train_result[:, 1]
    train_accuracy = roc_auc_score(temp_labels, train_result)
    log("n_examples %i%%, train accuracy: %0.4f, test accuracy: %0.4f" % (n_examples, train_accuracy, scores.mean()), bcolors.OKBLUE)

    results.append([n_examples, train_accuracy, scores.mean()])

    with open(results_file_name, 'w') as fp:
        a = csv.writer(fp, delimiter=',', lineterminator='\n')
        a.writerows(results)


log("done")
