############################################################################
# Create training and validation dataset for each chromosome
# by converting bigWig files to .h5. Note that we will also convert the data
# from 1bp resolution to 25 bp
############################################################################
from __future__ import print_function

import os
import sys
import pyBigWig
import numpy as np
import h5py
import argparse

from utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chrom", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    print("converting training data...", file=sys.stdout)
    output_filename = os.path.join(training_data_loc, "{}.h5".format(args.chrom))
    out_f = h5py.File(output_filename, "w")

    f = open(training_data_csv)
    f.readline()
    for line in f.readlines():
        ll = line.strip().split("\t")
        cell_type = ll[1]
        assay_type = ll[4]
        filename = ll[-1]

        input_filename = os.path.join("/hpcwork/izkf/projects/ENCODEImputation/data/training_data", filename)
        bw = pyBigWig.open(input_filename)
        data = bw.values(args.chrom, 0, bw.chroms()[args.chrom], numpy=True)

        m = data.shape[0] // 25
        y = np.zeros(m)

        for i in range(m):
            y[i] = np.mean(data[i * 25:(i + 1) * 25])

        out_f.create_dataset(name="{}.{}".format(cell_type, assay_type), data=y, dtype=np.float32)

    out_f.close()

    print("converting validation data...", file=sys.stdout)
    output_filename = os.path.join(validation_data_loc, "{}.h5".format(args.chrom))
    out_f = h5py.File(output_filename, "w")

    f = open(validation_data_csv)
    f.readline()
    for line in f.readlines():
        ll = line.strip().split("\t")
        cell_type = ll[1]
        assay_type = ll[4]
        filename = ll[-1]

        input_filename = os.path.join("/hpcwork/izkf/projects/ENCODEImputation/data/validation_data", filename)
        bw = pyBigWig.open(input_filename)
        data = bw.values(args.chrom, 0, bw.chroms()[args.chrom], numpy=True)

        m = data.shape[0] // 25
        y = np.zeros(m)

        for i in range(m):
            y[i] = np.mean(data[i * 25:(i + 1) * 25])

        out_f.create_dataset(name="{}.{}".format(cell_type, assay_type), data=y, dtype=np.float32)

    out_f.close()


if __name__ == '__main__':
    main()
