import cPickle
import matplotlib.pyplot as plt
import pandas as pd
from utils import *

# file name or "latest"
csv_file_name = "small_score per max_features. gbm. 255 factors2015-06-03 23-02-20"

if csv_file_name == "latest":
    csv_file_name = cPickle.load(open(results_dir + "latest_result.txt"))
else:
    csv_file_name = results_dir + csv_file_name + '.csv'

results = pd.read_csv(csv_file_name, header=None).values

x = results[:, 0]
train_accuracy = results[:, 2]
test_accuracy = results[:, 2]

plt.plot(x, train_accuracy, 'b', x, test_accuracy, 'r')
#plt.savefig(results_dir + csv_file_name + '.png', bbox_inches='tight')
plt.show()