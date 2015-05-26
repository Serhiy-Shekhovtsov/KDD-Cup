import matplotlib.pyplot as plt
import pandas as pd
from utils import *

csv_file_name = "find best n_factors(sparsesvd). max_features=.5, subsample=.9 2015-05-25 23-19-15"

results = pd.read_csv(results_dir + csv_file_name + '.csv', header=None).values

x = results[:, 0]
train_accuracy = results[:, 2]
test_accuracy = results[:, 2]

plt.plot(x, train_accuracy, 'b', x, test_accuracy, 'r')
#plt.savefig(results_dir + csv_file_name + '.png', bbox_inches='tight')
plt.show()