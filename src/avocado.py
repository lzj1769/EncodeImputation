##################################################################################
# This model is modified from
# https://github.com/jmschrei/avocado/blob/master/Avocado%20Training%20Demo.ipynb
# We are going to train it from scratch
##################################################################################
from __future__ import print_function

import os
import numpy as np
import h5py
from keras.models import Model, Sequential
from keras.layers import Input, Embedding, Flatten, Dense, concatenate
from keras.layers import Dropout
import argparse

from utils import *
from configure import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chrom", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=500)

    return parser.parse_args()


class DataGenerator(Sequential):
    'Generates data for Keras'

    def __init__(self, cell_types, assays, chrom_size, batch_size, data, window_size=25, shuffle=True):
        super(DataGenerator, self).__init__()
        self.cell_types = cell_types
        self.assays = assays
        self.chrom_size = chrom_size
        self.batch_size = batch_size
        self.data = data
        self.window_size = window_size
        self.shuffle = shuffle
        self.n_positions = chrom_size // window_size
        self.cell_assays = data.keys()
        self.indexes = np.arange(len(self.cell_assays) * self.n_positions)

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        celltype_idxs = np.zeros(len(indexes), dtype='int32')
        assay_idxs = np.zeros(len(indexes), dtype='int32')
        genomic_25bp_idxs = np.zeros(len(indexes), dtype='int32')
        y = np.zeros(len(indexes), dtype=np.float32)

        for i, index in enumerate(indexes):
            cell_assays_index = index // self.n_positions
            position_index = index - cell_assays_index * self.n_positions

            cell_assays = self.cell_assays[cell_assays_index]

            cell, assay = cell_assays.split(".")

            celltype_idxs[i] = self.cell_types.index(cell)
            assay_idxs[i] = self.assays.index(cell)
            genomic_25bp_idxs[i] = position_index

            y[i] = self.data[cell_assays][position_index]

        genomic_250bp_idxs = genomic_25bp_idxs // 10
        genomic_5kbp_idxs = genomic_25bp_idxs // 200

        x = {
            'celltype_input': celltype_idxs,
            'assay_input': assay_idxs,
            'genome_25bp_input': genomic_25bp_idxs,
            'genome_250bp_input': genomic_250bp_idxs,
            'genome_5kbp_input': genomic_5kbp_idxs
        }

        return x, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.cell_assays) * self.n_positions)
        if self.shuffle:
            np.random.shuffle(self.indexes)


#
# def data_generator(celltypes, assays, data, n_positions, batch_size):
#     while True:
#         celltype_idxs = np.zeros(batch_size, dtype='int32')
#         assay_idxs = np.zeros(batch_size, dtype='int32')
#         genomic_25bp_idxs = np.random.randint(n_positions, size=batch_size)
#         genomic_250bp_idxs = genomic_25bp_idxs // 10
#         genomic_5kbp_idxs = genomic_25bp_idxs // 200
#         value = np.zeros(batch_size)
#
#         keys = data.keys()
#         idxs = np.random.randint(len(data), size=batch_size)
#
#         for i, idx in enumerate(idxs):
#             celltype, assay = keys[idx]
#             track = data[(celltype, assay)]
#
#             celltype_idxs[i] = celltypes.index(celltype)
#             assay_idxs[i] = assays.index(assay)
#             value[i] = track[genomic_25bp_idxs[i]]
#
#         d = {
#             'celltype_input': celltype_idxs,
#             'assay_input': assay_idxs,
#             'genome_25bp_input': genomic_25bp_idxs,
#             'genome_250bp_input': genomic_250bp_idxs,
#             'genome_5kbp_input': genomic_5kbp_idxs
#         }
#
#         yield d, value


def build_model(n_celltypes, n_assays, n_genomic_positions,
                n_celltype_factors=25,
                n_assay_factors=25,
                n_25bp_factors=25,
                n_250bp_factors=30,
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


def main():
    args = parse_args()

    assert args.chrom is not None, "please choose a chromosome..."

    cell_types, assays = get_cell_assays()

    assays = assays[:2]

    chrom_size = chrom_size_dict[args.chrom]

    model = build_model(n_celltypes=len(cell_types),
                        n_assays=len(assays),
                        n_genomic_positions=chrom_size // 25)

    model.compile(optimizer="adam", loss="mse")
    model.summary()

    input_filename = os.path.join(training_data_loc, "{}.h5".format(args.chrom))
    data = h5py.File(input_filename, 'r')

    data_generator = DataGenerator(cell_types=cell_types,
                                   assays=assays,
                                   chrom_size=chrom_size,
                                   batch_size=args.batch_size,
                                   data=data)

    history = model.fit_generator(generator=data_generator,
                                  verbose=2,
                                  epochs=args.epochs,
                                  use_multiprocessing=True,
                                  workers=10,
                                  max_queue_size=20)


if __name__ == '__main__':
    main()
