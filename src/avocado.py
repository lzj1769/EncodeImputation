##################################################################################
# This model is modified from
# https://github.com/jmschrei/avocado/blob/master/Avocado%20Training%20Demo.ipynb
# We are going to train it from scratch
##################################################################################
from __future__ import print_function

import os
import numpy as np
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dense, concatenate
from keras.layers import Dropout
from keras.utils import plot_model
import argparse
import itertools

from utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4096)

    return parser.parse_args()


def data_generator(celltypes, assays, data, n_positions, batch_size):
    while True:
        celltype_idxs = np.zeros(batch_size, dtype='int32')
        assay_idxs = np.zeros(batch_size, dtype='int32')
        genomic_25bp_idxs = np.random.randint(n_positions, size=batch_size)
        genomic_250bp_idxs = genomic_25bp_idxs // 10
        genomic_5kbp_idxs = genomic_25bp_idxs // 200
        value = np.zeros(batch_size)

        keys = data.keys()
        idxs = np.random.randint(len(data), size=batch_size)

        for i, idx in enumerate(idxs):
            celltype, assay = keys[idx]
            track = data[(celltype, assay)]

            celltype_idxs[i] = celltypes.index(celltype)
            assay_idxs[i] = assays.index(assay)
            value[i] = track[genomic_25bp_idxs[i]]

        d = {
            'celltype_input': celltype_idxs,
            'assay_input': assay_idxs,
            'genome_25bp_input': genomic_25bp_idxs,
            'genome_250bp_input': genomic_250bp_idxs,
            'genome_5kbp_input': genomic_5kbp_idxs
        }

        yield d, value


def build_model(n_celltypes, n_assays, n_genomic_positions,
                n_celltype_factors=25,
                n_assay_factors=25,
                n_25bp_factors=25,
                n_250bp_factors=40,
                n_5kbp_factors=45,
                n_layers=2,
                n_nodes=2048):
    celltype_input = Input(shape=(1,), name="celltype_input")
    celltype_embedding = Embedding(n_celltypes, n_celltype_factors,
                                   input_length=1, name="celltype_embedding")
    celltype = Flatten()(celltype_embedding(celltype_input))

    assay_input = Input(shape=(1,), name="assay_input")
    assay_embedding = Embedding(n_assays, n_assay_factors,
                                input_length=1, name="assay_embedding")
    assay = Flatten()(assay_embedding(assay_input))

    genome_25bp_input = Input(shape=(1,), name="genome_25bp_input")
    genome_25bp_embedding = Embedding(n_genomic_positions, n_25bp_factors,
                                      input_length=1, name="genome_25bp_embedding")
    genome_25bp = Flatten()(genome_25bp_embedding(genome_25bp_input))

    genome_250bp_input = Input(shape=(1,), name="genome_250bp_input")
    genome_250bp_embedding = Embedding((n_genomic_positions / 10) + 1,
                                       n_250bp_factors, input_length=1, name="genome_250bp_embedding")
    genome_250bp = Flatten()(genome_250bp_embedding(genome_250bp_input))

    genome_5kbp_input = Input(shape=(1,), name="genome_5kbp_input")
    genome_5kbp_embedding = Embedding((n_genomic_positions / 200) + 1,
                                      n_5kbp_factors, input_length=1, name="genome_5kbp_embedding")
    genome_5kbp = Flatten()(genome_5kbp_embedding(genome_5kbp_input))

    layers = [celltype, assay, genome_25bp, genome_250bp, genome_5kbp]
    inputs = (celltype_input, assay_input, genome_25bp_input,
              genome_250bp_input, genome_5kbp_input)

    x = concatenate(layers)
    for i in range(n_layers):
        x = Dense(n_nodes, activation='relu', name="dense_{}".format(i))(x)
        x = Dropout(0.5)(x)

    outputs = Dense(1, name="y_pred")(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model


def get_cell_assays():
    cell_type = []
    assays = []

    f = open(training_data_csv)
    f.readline()

    for line in f.readlines():
        ll = line.strip().split("\t")
        cell_type.append(ll[1])
        assays.append(ll[4])

    f.close()

    return list(set(cell_type)), list(set(assays))


def get_data(celltypes, assays, chrom):
    data = dict()

    for celltype, assay in itertools.product(celltypes, assays):
        filename = os.path.join(training_data_loc, assay, "{}.{}.{}.npz".format(celltype, assay, chrom))

        if os.path.exists(filename):
            print("loading data from {}...".format(filename))
            data[(celltype, assay)] = np.load(filename)['data']

    return data


def main():
    celltypes, assays = get_cell_assays()

    for chrom in chrom_list:
        chrom_size = chrom_size_dict[chrom]

        model = build_model(n_celltypes=len(celltypes),
                            n_assays=len(assays),
                            n_genomic_positions=chrom_size)

        model.compile(optimizer="adam", loss="mse")

        data = get_data(celltypes=celltypes,
                        assays=assays,
                        chrom=chrom)

        x_train = data_generator(celltypes=celltypes,
                                 assays=assays,
                                 data=data,
                                 n_positions=chrom_size,
                                 batch_size=4096)

        history = model.fit_generator(generator=x_train,
                                      verbose=1,
                                      epochs=2)

        exit(0)


if __name__ == '__main__':
    main()
