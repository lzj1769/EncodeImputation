import os
import argparse
import numpy as np
import pathlib

training_data_tsv = "/hpcwork/izkf/projects/ENCODEImputation/local/TSV/metadata_training_data.tsv"
validation_data_tsv = "/hpcwork/izkf/projects/ENCODEImputation/local/TSV/metadata_validation_data.tsv"

chrom_size_dict = {'chr1': 9958247,
                   'chr2': 9687698,
                   'chr3': 7931798,
                   'chr4': 7608557,
                   'chr5': 7261460,
                   'chr6': 6832240,
                   'chr7': 6373839,
                   'chr8': 5805546,
                   'chr9': 5535789,
                   'chr10': 5351815,
                   'chr11': 5403465,
                   'chr12': 5331013,
                   'chr13': 4574574,
                   'chr14': 4281749,
                   'chr15': 4079648,
                   'chr16': 3613240,
                   'chr17': 3330298,
                   'chr18': 3214932,
                   'chr19': 2344705,
                   'chr20': 2577636,
                   'chr21': 1868374,
                   'chr22': 2032739,
                   'chrX': 6241636}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-chr", "--chrom", type=str, default=None)
    parser.add_argument("-i", "--input_loc", type=str, default=None)
    parser.add_argument("-o", "--output_loc", type=str, default=None)
    parser.add_argument("-s", "--seed", type=int, default=2019)
    parser.add_argument("-v", "--verbose", type=int, default=0)

    return parser.parse_args()


def get_cells_assays():
    cells = []
    assays = []

    f = open(training_data_tsv)
    f.readline()

    for line in f.readlines():
        ll = line.strip().split("\t")
        if ll[1] not in cells:
            cells.append(ll[1])

        if ll[2] not in assays:
            assays.append(ll[2])
    f.close()

    return cells, assays


def main():
    args = parse_args()

    assert args.chrom is not None, "please choose a chromosome..."

    cells, assays = get_cells_assays()

    pathlib.Path(args.output_loc).mkdir(parents=True, exist_ok=True)

    cell_assay_indexes = []
    y = np.empty(0, dtype=np.float32)
    for i, cell in enumerate(cells):
        for j, assay in enumerate(assays):
            filename = os.path.join(args.input_loc, "{}{}.npy".format(cell, assay))
            if os.path.exists(filename):
                data = np.arcsinh(np.load(filename)[()][args.chrom])
                y = np.append(y, data)
                cell_assay_indexes.append((i, j))

    length = len(cell_assay_indexes) * chrom_size_dict[args.chrom]

    x = np.empty((length, 5), dtype=np.int32)
    for i in range(length):
        cell_assay_index = i // chrom_size_dict[args.chrom]
        genomic_25bp_index = i - cell_assay_index * chrom_size_dict[args.chrom]
        genomic_250bp_index = genomic_25bp_index // 10
        genomic_5kbp_index = genomic_25bp_index // 200

        cell_index, assay_index = cell_assay_indexes[cell_assay_index]
        x[i] = np.array([cell_index, assay_index, genomic_25bp_index,
                         genomic_250bp_index, genomic_5kbp_index])

    output_filename_x = os.path.join(args.output_loc, "{}_x".format(args.chrom))
    output_filename_y = os.path.join(args.output_loc, "{}_y".format(args.chrom))
    np.save(output_filename_x, x)
    np.save(output_filename_y, y)


if __name__ == '__main__':
    main()
