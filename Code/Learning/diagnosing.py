import cPickle
import csv
import numpy as np
from numpy import squeeze
import pandas as pd
from sklearn import svm, cross_validation, preprocessing, ensemble
from sklearn.decomposition import TruncatedSVD
from utils import *


log("find best decomposition for appetency")

data = cPickle.load(open(clean_data_dir + "train_val_set.csc"))
train_indices = np.load(clean_data_dir + "train_indices.npy")
train_labels = pd.read_csv(labels_dir + 'orange_large_train_appetency.labels', header=None)
train_labels = squeeze(train_labels.values)[train_indices]

results_file_name = results_dir + datetime.now().strftime('%Y-%m-%d %H-%M-%S') + ".csv"
results = []

algorithms = ["GBM"]

for n_factors in range(10, 300, 10):
    for alg in algorithms:
        log("*** n_factors = %i ***" % n_factors)

        log("TruncatedSVD.fit_transform data")
        svd = TruncatedSVD(n_components=n_factors, random_state=42)
        train_data = svd.fit_transform(data)
        log("TruncatedSVD.fit_transform done")

        if (alg == "SVM 1") | (alg == "SVM 2"):
            train_data = preprocessing.scale(train_data)

        if alg == "GBM":
            clf = ensemble.GradientBoostingClassifier(random_state=0)
        elif alg == "SVM 1":
            clf = svm.SVC(kernel='poly', degree=1)
        elif alg == "SVM 2":
            clf = svm.SVC(kernel='poly', degree=2)

        log("cross_val_score, alg = " + alg)
        scores = cross_validation.cross_val_score(clf, train_data, train_labels, cv=3,
                                                  verbose=4, scoring='roc_auc')
        log("n_factors %i, algorithm %s. accuracy: %0.2f" % (n_factors, alg, scores.mean()), bcolors.OKBLUE)

        results.append([n_factors, alg, scores.mean()])

        with open(results_file_name, 'w') as fp:
            a = csv.writer(fp, delimiter=',', lineterminator='\n')
            a.writerows(results)


log("done")
