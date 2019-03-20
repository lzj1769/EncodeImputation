##########################################
# Convert bigWig files to .npz
##########################################
from __future__ import print_function

import os
import sys
import pyBigWig
import numpy as np

from utils import *

input_filename = sys.argv[1]
output_location = sys.argv[2]
cell_type = sys.argv[3]
assay_type = sys.argv[4]


bw = pyBigWig.open(input_filename)

for chrom in chrom_list:
    data = bw.values(chrom, 0, bw.chroms()[chrom], numpy=True)
    output_filename = os.path.join(output_location, "{}.{}.{}.npz".format(cell_type, assay_type, chrom))
    np.savez(file=output_filename, data=data)
