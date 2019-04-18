import os
import sys
import numpy as np
import torch
from torch import nn
import torch.optim as optimizers
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import argparse
import time
import random
from contextlib import contextmanager

training_data_csv = "/hpcwork/izkf/projects/ENCODEImputation/data/training_data/metadata_training_data.tsv"
training_data_loc = "/hpcwork/izkf/projects/ENCODEImputation/input/training_data"

validation_data_csv = "/hpcwork/izkf/projects/ENCODEImputation/data/validation_data/metadata_training_data.tsv"
validation_data_loc = "/hpcwork/izkf/projects/ENCODEImputation/input/validation_data"

chrom_size_dict = {'chr1': 248956422,
                   'chr2': 242193529,
                   'chr3': 198295559,
                   'chr4': 190214555,
                   'chr5': 181538259,
                   'chr6': 170805979,
                   'chr7': 159345973,
                   'chr8': 145138636,
                   'chr9': 138394717,
                   'chr10': 133797422,
                   'chr11': 135086622,
                   'chr12': 133275309,
                   'chr13': 114364328,
                   'chr14': 107043718,
                   'chr15': 101991189,
                   'chr16': 90338345,
                   'chr17': 83257441,
                   'chr18': 80373285,
                   'chr19': 58617616,
                   'chr20': 64444167,
                   'chr21': 46709983,
                   'chr22': 50818468,
                   'chrX': 156040895}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--chrom", type=str, default=None)
    parser.add_argument("-bs", "--batch_size", type=int, default=40960)
    parser.add_argument("-e", "--epochs", type=int, default=120)
    parser.add_argument("-s", "--seed", type=int, default=2019)
    parser.add_argument("-v", "--verbose", type=int, default=2)

    return parser.parse_args()


@contextmanager
def timer(msg):
    t0 = time.time()
    print(f'[{msg}] start.', file=sys.stdout)
    yield
    elapsed_time = time.time() - t0
    print(f'[{msg}] done in {elapsed_time / 60:.2f} min.', file=sys.stdout)


def get_cells_assays():
    cells = []
    assays = []

    f = open(training_data_csv)
    f.readline()

    for line in f.readlines():
        ll = line.strip().split("\t")
        cells.append(ll[1])
        assays.append(ll[2])

    f.close()

    return list(set(cells)), list(set(assays))


def seed_torch(seed=2019):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class Avocado(nn.Module):
    def __init__(self, n_cells, n_assays, n_positions_25bp, n_positions_250bp, n_positions_5kbp,
                 cell_embedding_dim=10, assay_embedding_dim=10, positions_25bp_embedding_dim=25,
                 positions_250bp_embedding_dim=25, positions_5kbp_embedding_dim=25, n_hidden_units=256):
        super(Avocado, self).__init__()

        # cell embedding matrix
        self.cell_embedding = nn.Embedding(num_embeddings=n_cells,
                                           embedding_dim=cell_embedding_dim)

        # assay embedding matrix
        self.assay_embedding = nn.Embedding(num_embeddings=n_assays,
                                            embedding_dim=assay_embedding_dim)

        # genomic positions embedding matrix
        self.positions_25bp_embedding = nn.Embedding(num_embeddings=n_positions_25bp,
                                                     embedding_dim=positions_25bp_embedding_dim)

        self.positions_250bp_embedding = nn.Embedding(num_embeddings=n_positions_250bp,
                                                      embedding_dim=positions_250bp_embedding_dim)

        self.positions_5kbp_embedding = nn.Embedding(num_embeddings=n_positions_5kbp,
                                                     embedding_dim=positions_5kbp_embedding_dim)

        in_features = cell_embedding_dim + assay_embedding_dim + positions_25bp_embedding_dim + \
                      positions_250bp_embedding_dim + positions_5kbp_embedding_dim

        self.linear1 = nn.Linear(in_features, n_hidden_units)
        self.linear2 = nn.Linear(n_hidden_units, n_hidden_units)
        self.linear_out = nn.Linear(n_hidden_units, 1)

    def forward(self, x):
        cell_index = x['cell_index']
        assay_index = x['assay_index']
        positions_25bp_index = x['positions_25bp_index']
        positions_250bp_index = x['positions_250bp_index']
        positions_5kbp_index = x['positions_5kbp_index']

        inputs = torch.cat((self.cell_embedding.weight[cell_index],
                            self.assay_embedding.weight[assay_index],
                            self.positions_25bp_embedding.weight[positions_25bp_index],
                            self.positions_250bp_embedding.weight[positions_250bp_index],
                            self.positions_5kbp_embedding.weight[positions_5kbp_index]), 1)

        x = F.relu(self.linear1(inputs))
        x = F.relu(self.linear2(x))
        out = self.linear_out(x)

        return out


class EncodeImputationDataset(Dataset):
    """Encode dataset."""

    def __init__(self, cells, assays, n_positions_25bp, chromosome, data_loc):
        self.cells = cells
        self.assays = assays
        self.n_cells = len(cells)
        self.n_assays = len(assays)
        self.n_positions_25bp = n_positions_25bp
        self.chromosome = chromosome
        self.data = np.empty(shape=(self.n_cells, self.n_assays, self.n_positions_25bp), dtype=np.float32)
        self.cell_assay_indexes = list()

        for i, cell in enumerate(self.cells):
            for j, assay in enumerate(self.assays):
                filename = os.path.join(data_loc, "{}{}.npz".format(cell, assay))
                if os.path.exists(filename):
                    with np.load(filename) as data:
                        self.data[i, j, :] = data[self.chromosome]

                    self.cell_assay_indexes.append((i, j))

        self.indexes = np.arange(len(self.cell_assay_indexes) * n_positions_25bp)

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index):
        cell_assay_index = index // self.n_positions_25bp
        genomic_25bp_index = index - cell_assay_index * self.n_positions_25bp
        genomic_250bp_index = genomic_25bp_index // 10
        genomic_5kbp_index = genomic_25bp_index // 200

        cell_index, assay_index = self.cell_assay_indexes[cell_assay_index]
        y = self.data[cell_index, assay_index, genomic_25bp_index]
        x = {
            'cell_index': cell_index,
            'assay_index': assay_index,
            'positions_25bp_index': genomic_25bp_index,
            'positions_250bp_index': genomic_250bp_index,
            'positions_5kbp_index': genomic_5kbp_index
        }

        return x, y


def get_positions(chrom_size):
    n_positions_25bp = chrom_size // 25

    if n_positions_25bp % 10 == 0:
        n_positions_250bp = n_positions_25bp // 10
    else:
        n_positions_250bp = n_positions_25bp // 10 + 1

    if n_positions_25bp % 200 == 0:
        n_positions_5kbp = n_positions_25bp // 200
    else:
        n_positions_5kbp = n_positions_25bp // 200 + 1

    return n_positions_25bp, n_positions_250bp, n_positions_5kbp


def update_embedding(model, embedding, x_batch, y_batch, optimizer, loss_fn):
    assert embedding not in ['cell_embedding', 'assay_embedding', 'positions_25bp_embedding',
                             'positions_250bp_embedding',
                             'positions_5kbp_embedding'], "embedding  {} doesn't exist".format(embedding)

    if embedding == 'cell_embedding':
        model.cell_embedding.weight.requires_grad = True
        model.assay_embedding.weight.requires_grad = False
        model.positions_25bp_embedding.weight.requires_grad = False
        model.positions_250bp_embedding.weight.requires_grad = False
        model.positions_5kbp_embedding.weight.requires_grad = False

    elif embedding == 'assay_embedding':
        model.cell_embedding.weight.requires_grad = False
        model.assay_embedding.weight.requires_grad = True
        model.positions_25bp_embedding.weight.requires_grad = False
        model.positions_250bp_embedding.weight.requires_grad = False
        model.positions_5kbp_embedding.weight.requires_grad = False

    elif embedding == 'positions_25bp_embedding':
        model.cell_embedding.weight.requires_grad = False
        model.assay_embedding.weight.requires_grad = False
        model.positions_25bp_embedding.weight.requires_grad = True
        model.positions_250bp_embedding.weight.requires_grad = False
        model.positions_5kbp_embedding.weight.requires_grad = False

    elif embedding == 'positions_250bp_embedding':
        model.cell_embedding.weight.requires_grad = False
        model.assay_embedding.weight.requires_grad = False
        model.positions_25bp_embedding.weight.requires_grad = False
        model.positions_250bp_embedding.weight.requires_grad = True
        model.positions_5kbp_embedding.weight.requires_grad = False

    elif embedding == 'positions_5kbp_embedding':
        model.cell_embedding.weight.requires_grad = False
        model.assay_embedding.weight.requires_grad = False
        model.positions_25bp_embedding.weight.requires_grad = False
        model.positions_250bp_embedding.weight.requires_grad = False
        model.positions_5kbp_embedding.weight.requires_grad = True

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    y_pred = model(x_batch)
    loss = loss_fn(y_batch, y_pred)
    loss.backward()
    optimizer.step()

    return loss.item()


def main():
    args = parse_args()

    assert args.chrom is not None, "please choose a chromosome..."

    seed_torch(seed=args.seed)

    cells, assays = get_cells_assays()

    with timer("build model"):
        chrom_size = chrom_size_dict[args.chrom]

        n_positions_25bp, n_positions_250bp, n_positions_5kbp = get_positions(chrom_size)

        avocado = Avocado(n_cells=len(cells), n_assays=len(assays),
                          n_positions_25bp=n_positions_25bp,
                          n_positions_250bp=n_positions_250bp,
                          n_positions_5kbp=n_positions_5kbp)

    with timer('load data'):
        train_dataset = EncodeImputationDataset(cells=cells, assays=assays, n_positions_25bp=n_positions_25bp,
                                                chromosome=args.chrom, data_loc=training_data_loc)

        valid_dataset = EncodeImputationDataset(cells=cells, assays=assays, n_positions_25bp=n_positions_25bp,
                                                chromosome=args.chrom, data_loc=validation_data_loc)

        train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=args.batch_size)
        valid_dataloader = DataLoader(dataset=valid_dataset, shuffle=False, batch_size=args.batch_size)

    with timer('training'):
        loss = nn.MSELoss()

        for epoch in range(args.epochs):
            sgd = optimizers.SGD(avocado.parameters(), lr=0.001, momentum=0.9)

            # training
            running_loss = 0.0
            for i, data in enumerate(train_dataloader):
                # get the inputs
                x, y = data

                # iteratively update embedding matrix for cells, assays and genomic positions
                running_loss += update_embedding(model=avocado, embedding='cell_embedding', x_batch=x, y_batch=y,
                                                 optimizer=sgd, loss_fn=loss)

                running_loss += update_embedding(model=avocado, embedding='assay_embedding', x_batch=x, y_batch=y,
                                                 optimizer=sgd, loss_fn=loss)

                running_loss += update_embedding(model=avocado, embedding='positions_25bp_embedding', x_batch=x,
                                                 y_batch=y,
                                                 optimizer=sgd, loss_fn=loss)

                running_loss += update_embedding(model=avocado, embedding='positions_250bp_embedding', x_batch=x,
                                                 y_batch=y,
                                                 optimizer=sgd, loss_fn=loss)

                running_loss += update_embedding(model=avocado, embedding='positions_5kbp_embedding', x_batch=x,
                                                 y_batch=y,
                                                 optimizer=sgd, loss_fn=loss)

                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

            # # validation
            # valid_loss = 0
            # for i, data in enumerate(valid_dataloader):


if __name__ == '__main__':
    main()
