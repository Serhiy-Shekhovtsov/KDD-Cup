import cPickle
import csv
import numpy as np
from numpy import squeeze
import pandas as pd
from sklearn import svm, cross_validation, preprocessing, ensemble
from sklearn.decomposition import TruncatedSVD
from utils import *


log("decompose and save to .npy file")

n_factors = 200

log("reading data")
data = cPickle.load(open(clean_data_dir + "train.csc"))
train_indices = np.load(clean_data_dir + "train_indices.npy")
train_labels = pd.read_csv(labels_dir + 'orange_large_train_appetency.labels', header=None)
train_labels = squeeze(train_labels.values)[train_indices]

log("TruncatedSVD.fit_transform data")
svd = TruncatedSVD(n_components=n_factors, random_state=42)
transformed_data = svd.fit_transform(data)
log("TruncatedSVD.fit_transform done")

cPickle.dump(transformed_data, clean_data_dir + "train_indices.npy")

log("done")
