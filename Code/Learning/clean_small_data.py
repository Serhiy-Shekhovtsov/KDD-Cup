import scipy
import cPickle
import pandas as pd
import numpy as np
from scipy.sparse import hstack, vstack
from sklearn import preprocessing
from sympy.matrices import sparse
from utils import *

log("read text data")
data = pd.read_csv(src_dir + "orange_small_train.data", sep='\t', header=0)  # , dtype=object

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

data = data.loc[:, data.isnull().sum() < data.shape[0]]

log("split data")

# split to numeric and non numeric
numeric_data = data.select_dtypes(include=numerics)
not_num_data = data.select_dtypes(exclude=numerics)

log("fix num data")

# replace NaN in numeric data
numeric_data = numeric_data.fillna(numeric_data.mean())

# scale data
numeric_data = preprocessing.scale(numeric_data)

del data, numeric_data

log("fix not num data")

# replace NaN
not_num_data = not_num_data.fillna("ValueMissing")

# binarization
full_data = None

for i in range(0, not_num_data.shape[1]):
    log("column " + str(i))

    enc = preprocessing.LabelEncoder()
    col_data = enc.fit_transform(not_num_data.iloc[:, i])

    enc = preprocessing.OneHotEncoder()
    col_data = enc.fit_transform(col_data)
    # full_data = col_data if full_data is None else hstack([full_data, col_data])
    # full_data = scipy.sparse.csc_matrix(full_data)

log("not num data fixed!")

log("save")
cPickle.dump(full_data, open(clean_data_dir + "small_data_full.csc", "w"))

log("done")