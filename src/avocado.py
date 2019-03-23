##################################################################################
# This model is modified from
# https://github.com/jmschrei/avocado/blob/master/Avocado%20Training%20Demo.ipynb
# We are going to train it from scratch
##################################################################################
from __future__ import print_function

import os
import sys
import numpy as np
import h5py
import argparse
import warnings

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dense, concatenate
from keras.layers import Dropout
from keras.utils import Sequence
from keras.utils.io_utils import h5dict
from keras.callbacks import Callback
from keras.engine.saving import _serialize_model
from keras.utils.multi_gpu_utils import multi_gpu_model

from utils import *
from configure import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--chrom", type=str, default=None)
    parser.add_argument("-bs", "--batch_size", type=int, default=40960)
    parser.add_argument("-e", "--epochs", type=int, default=25)
    parser.add_argument("-v", "--verbose", type=int, default=2)

    return parser.parse_args()


class DataGenerator(Sequence):
    'Generates data for Keras'

    def __init__(self, cell_types, assays, n_positions, batch_size, data, shuffle=True):
        super(DataGenerator, self).__init__()
        self.cell_types = cell_types
        self.assays = assays
        self.n_positions = n_positions
        self.batch_size = batch_size
        self.data = data
        self.shuffle = shuffle
        self.cell_assays = self.data.keys()
        self.indexes = np.arange(len(self.cell_assays) * self.n_positions)

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        celltype_idxs = np.zeros(len(indexes), dtype=np.int32)
        assay_idxs = np.zeros(len(indexes), dtype=np.int32)
        genomic_25bp_idxs = np.zeros(len(indexes), dtype=np.int32)

        y = np.zeros(len(indexes), dtype=np.float32)

        for i, index in enumerate(indexes):
            cell_assays_index = index // self.n_positions
            position_index = index - cell_assays_index * self.n_positions

            cell_assays = self.cell_assays[cell_assays_index]

            cell, assay = cell_assays.split(".")

            celltype_idxs[i] = self.cell_types.index(cell)
            assay_idxs[i] = self.assays.index(assay)
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


class MultiGPUModelCheckpoint(Callback):
    """Additional keyword args are passed to ModelCheckpoint; see those docs for information on what args are accepted.
    https://github.com/TextpertAi/alt-model-checkpoint/blob/master/alt_model_checkpoint/__init__.py
    """

    def __init__(self,
                 filepath=None,
                 model_to_save=None,
                 best=np.Inf,
                 monitor='val_loss'):
        super(MultiGPUModelCheckpoint, self).__init__()
        self.filepath = filepath
        self.model_to_save = model_to_save
        self.best = best
        self.monitor = monitor
        self.monitor_op = np.less

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        filepath = self.filepath.format(epoch=epoch + 1, **logs)
        current = logs.get(self.monitor)

        if current is None:
            warnings.warn('Can save best model only with %s available, '
                          'skipping.' % self.monitor, RuntimeWarning)
        else:
            if self.monitor_op(current, self.best):
                print('\nEpoch %05d: %s improved from %0.8f to %0.8f,'
                      ' saving model to %s'
                      % (epoch + 1, self.monitor, self.best,
                         current, filepath), file=sys.stdout)
                self.best = current

                if os.path.exists(filepath):
                    os.remove(filepath)

                try:
                    f = h5dict(filepath, mode='w')
                    _serialize_model(self.model_to_save, f, include_optimizer=True)
                    f.close()
                except:
                    print("There is something wrong with saving model, will skip it", file=sys.stderr)

            else:
                print('\nEpoch %05d: %s did not improve from %0.8f' %
                      (epoch + 1, self.monitor, self.best), file=sys.stdout)


def build_model(n_celltypes, n_assays, n_genomic_positions,
                n_celltype_factors=10,
                n_assay_factors=10,
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
    inputs = (celltype_input, assay_input, genome_25bp_input, genome_250bp_input, genome_5kbp_input)

    x = concatenate(layers)
    for i in range(n_layers):
        x = Dense(n_nodes, activation='relu', name="dense_{}".format(i))(x)
        x = Dropout(0.5)(x)

    outputs = Dense(1, activation='relu', name="y_pred")(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model


def main():
    args = parse_args()

    assert args.chrom is not None, "please choose a chromosome..."

    cell_types, assays = get_train_cell_assays()

    chrom_size = chrom_size_dict[args.chrom]
    n_genomic_positions = chrom_size // 25

    print("load training and validation data...", file=sys.stderr)
    print("===========================================================================\n", file=sys.stderr)

    input_filename = os.path.join(training_data_loc, "{}.h5".format(args.chrom))
    training_data = dict()
    with h5py.File(input_filename, 'r') as f:
        for key in f.iterkeys():
            training_data[key] = f[key][:]

    input_filename = os.path.join(validation_data_loc, "{}.h5".format(args.chrom))
    validation_data = dict()
    with h5py.File(input_filename, 'r') as f:
        for key in f.iterkeys():
            validation_data[key] = f[key][:]

    train_generator = DataGenerator(cell_types=cell_types,
                                    assays=assays,
                                    n_positions=n_genomic_positions,
                                    batch_size=args.batch_size,
                                    data=training_data,
                                    shuffle=True)

    valid_generator = DataGenerator(cell_types=cell_types,
                                    assays=assays,
                                    n_positions=n_genomic_positions,
                                    batch_size=args.batch_size,
                                    data=validation_data,
                                    shuffle=False)

    model = build_model(n_celltypes=len(cell_types),
                        n_assays=len(assays),
                        n_genomic_positions=n_genomic_positions)

    parallel_model = multi_gpu_model(model=model, gpus=2, cpu_merge=False)
    parallel_model.compile(optimizer="adam", loss="mse")
    model.compile(optimizer="adam", loss="mse")

    model_filename = os.path.join(model_loc, "avocado.{}.h5".format(args.chrom))
    model_checkpoint = MultiGPUModelCheckpoint(filepath=model_filename,
                                               model_to_save=model)

    history = parallel_model.fit_generator(generator=train_generator,
                                           validation_data=valid_generator,
                                           verbose=args.verbose,
                                           epochs=args.epochs,
                                           use_multiprocessing=True,
                                           workers=48,
                                           max_queue_size=1000,
                                           callbacks=[model_checkpoint])

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    output_filename = os.path.join("/home/rs619065/EncodeImputation/vis", "avocado.{}.pdf".format(args.chrom))
    plt.savefig(output_filename)

    print("complete!!", file=sys.stdout)


if __name__ == '__main__':
    main()
