import scipy
import cPickle
from scipy.io import mmread, mmwrite
from scipy.sparse import vstack
from sympy.matrices import sparse
from utils import *

# combine chunks of sparsed matrix
for data_type in data_types:
    log("transforming %s data" % data_type)
    full_data = None
    for chunk in data_chunks:
        log("reading chunk %i" % chunk)
        for part in data_parts:
            log("  part" + str(part))
            data = mmread(clean_data_dir + "test.%i_%i.mm" % (chunk, part))

            full_data = data if full_data is None else vstack([full_data, data])
            del data
        log("chunk %i reading completed" % chunk)

    log("saving %s data" % data_type)
    smat = scipy.sparse.csc_matrix(full_data)
    cPickle.dump(smat, open(clean_data_dir + data_type + ".csc", "w"))

    log("saving done")