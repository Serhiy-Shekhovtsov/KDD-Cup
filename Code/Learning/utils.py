from datetime import datetime

data_dir = '../../Data/'
clean_data_dir = data_dir + 'clean/'
labels_dir = clean_data_dir + 'labels/'
svd_dir = clean_data_dir + 'svd/'
results_dir = data_dir + 'results/'
src_dir = data_dir + 'src/'

data_types = ["test", "train"]
data_chunks = [1, 2, 3, 4, 5]
data_parts = [1, 2, 3, 4]

small = True
data_size = "small_" if small else ""


def get_chunks(l, n):
    avg = len(l) / float(n)
    out = []
    last = 0.0

    while last < len(l):
        out.append(l[int(last):int(last + avg)])
        last += avg

    return out


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
