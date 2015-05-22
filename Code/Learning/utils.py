from datetime import datetime

data_dir = '../../Data/'
clean_data_dir = data_dir + 'clean/'
labels_dir = clean_data_dir + 'labels/'
results_dir = data_dir + 'results/'

data_types = ["test", "train"]
data_chunks = [1, 2, 3, 4, 5]
data_parts = [1, 2, 3, 4]


def log(message, color=None):
    if not color:
        print "at " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " " + str(message)
    else:
        print color + " at " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " " + str(message) + bcolors.ENDC


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
