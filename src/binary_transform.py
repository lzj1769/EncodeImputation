import sys
import os
import argparse
import numpy


def parse_arguments():
    parser = argparse.ArgumentParser(
        prog='ENCODE Imputation Challenge scoring script')
    parser.add_argument('raw',
                        help='NPY file with raw data')
    parser.add_argument('--out-npy-prefix',
                        help='Output prefix for .npy or .npz')
    args = parser.parse_args()
    args.chrom = ['chr' + str(i) for i in range(1, 23)] + ['chrX']
    return args


def main():
    args = parse_arguments()

    raw_dict = numpy.load(args.raw, allow_pickle=True)[()]

    for chrom in args.chrom:
        raw_dict[chrom] = numpy.arcsinh(raw_dict[chrom])

    numpy.save(args.out_npy_prefix, raw_dict)


if __name__ == '__main__':
    main()
