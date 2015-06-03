import cPickle
import os.path
import csv
from sparsesvd import sparsesvd
import numpy as np
from numpy import squeeze
import pandas as pd
from sklearn import svm, cross_validation, preprocessing, ensemble
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import safe_sparse_dot
from utils import *


def load_decomposed(n_factors, alg="TruncatedSVD"):
    log("load_decomposed started")

    file_name = svd_dir + data_size + str(n_factors) + "_factors"
    if alg == "sparsesvd":
        file_name += "(sparsesvd)"
    file_name += ".npy"

    if os.path.isfile(file_name):
        log("load decomposed data from file")
        return cPickle.load(open(file_name))

    log("decompose and save to .npy file")

    log("reading data files")
    full_data = cPickle.load(open(clean_data_dir + data_size + "full_data.csc"))
    data = cPickle.load(open(clean_data_dir + data_size + "train_val_set.csc"))

    transformed_data = None

    log("decompose")
    if alg == "TruncatedSVD":
        svd = TruncatedSVD(n_components=n_factors, random_state=42)
        svd.fit(full_data)

        transformed_data = svd.transform(data)
    elif alg == "sparsesvd":
        ut, s, vt = sparsesvd(full_data, n_factors)

        transformed_data = safe_sparse_dot(data, vt.T)

    log("decompose completed")

    log("save and return")

    cPickle.dump(transformed_data, open(file_name, 'w'))

    return transformed_data
