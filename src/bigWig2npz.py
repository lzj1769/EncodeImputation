##########################################
# Convert bigWig files to .npz
##########################################
from __future__ import print_function

import os
import sys
import pyBigWig
import numpy as np

input_filename = sys.argv[1]
output_location = sys.argv[2]
cell_type = sys.argv[3]
assay_type = sys.argv[4]

# We restrict this challenge to chromosomes 1-22 and
# chromosome X (i.e. ignore any data on chrY and chrM)
chrom_list = ['chr1', 'chr2', 'chr3',
              'chr4', 'chr5', 'chr6',
              'chr7', 'chr8', 'chr9',
              'chr10', 'chr11', 'chr12',
              'chr13', 'chr14', 'chr15',
              'chr16', 'chr17', 'chr18',
              'chr19', 'chr20', 'chr21',
              'chr22', 'chrX']

bw = pyBigWig.open(input_filename)

for chrom in chrom_list:
    data = bw.values(chrom, 0, bw.chroms()[chrom], numpy=True)
    output_filename = os.path.join(output_location, "{}.{}.{}.npz".format(cell_type, assay_type, chrom))
    np.savez(file=output_filename, data=data)
