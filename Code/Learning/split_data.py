import cPickle
import random
import numpy as np
from sklearn.decomposition import TruncatedSVD
from utils import *


log("read train matrix")
data = cPickle.load(open(clean_data_dir + "train" + ".csc"))
n_items = data.shape[0]
split = int(n_items * .8)
indices = np.arange(0, n_items)

random.seed(333)
random.shuffle(indices)

train_indices = indices[0:split]
test_indices = indices[split:]

cPickle.dump(data[train_indices, :], open(clean_data_dir + "train_val_set.csc", "w"))
cPickle.dump(data[test_indices, :], open(clean_data_dir + "test_set.csc", "w"))
np.save(clean_data_dir + "train_indices.npy", train_indices)
np.save(clean_data_dir + "test_indices.npy", test_indices)
log("done")
