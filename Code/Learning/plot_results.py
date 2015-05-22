import matplotlib.pyplot as plt
import pandas as pd
from utils import *

results = pd.read_csv(
    results_dir + 'accuracy per subsample. gbm. 200 factors2015-05-22 18-39-25.csv',
    header=None).values

x = results[:, 0]
train_accuracy = results[:, 1]
test_accuracy = results[:, 2]

plt.plot(x, train_accuracy, 'b', x, test_accuracy, 'r')
plt.show()