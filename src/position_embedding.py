import os
import argparse
import numpy as np
import random
import time
import warnings
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.optim as optimizers
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset

training_data_csv = "/hpcwork/izkf/projects/ENCODEImputation/data_true/training_data/metadata_training_data.tsv"
training_data_loc = "/hpcwork/izkf/projects/ENCODEImputation/npz_file/Train_log"

validation_data_csv = "/hpcwork/izkf/projects/ENCODEImputation/data_true/validation_data/metadata_training_data.tsv"
validation_data_loc = "/hpcwork/izkf/projects/ENCODEImputation/npz_file/Valid_log"

model_loc = "/hpcwork/izkf/projects/ENCODEImputation/model_li"
vis_loc = "/home/rs619065/EncodeImputation/vis"

chrom_size_dict = {'chr1': 9958247,
                   'chr2': 9687699,
                   'chr3': 7931798,
                   'chr4': 7608557,
                   'chr5': 7261461,
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
                   'chr16': 3613242,
                   'chr17': 3330298,
                   'chr18': 3214932,
                   'chr19': 2344705,
                   'chr20': 2577636,
                   'chr21': 1868374,
                   'chr22': 2032739,
                   'chrX': 6241636}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--chromosome", type=str, default=None,
                        help="The name of chromosome for training")
    parser.add_argument("-bs", "--batch_size", type=int, default=20480)
    parser.add_argument("-e", "--epochs", type=int, default=30)
    parser.add_argument("-s", "--seed", type=int, default=2019)
    parser.add_argument("-v", "--verbose", type=int, default=0)
    parser.add_argument("-w", "--num_workers", type=int, default=24)
    parser.add_argument("-r", "--resolution", type=int, default=0,
                        help="Which genomic position resolution (25bp, 250bp and 5kbp)"
                             "to use as embedding matrix. ")

    return parser.parse_args()


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


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class GenomicPositionEmbedding(nn.Module):
    def __init__(self, n_positions, n_output_units, embedding_dim=25, n_hidden_units=256):
        super(GenomicPositionEmbedding, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=n_positions, embedding_dim=embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, n_hidden_units)
        self.linear2 = nn.Linear(n_hidden_units, n_hidden_units)
        self.linear_out = nn.Linear(n_hidden_units, n_output_units)

    def forward(self, x):
        inputs = self.embedding.weight[x],
        x = F.relu(self.linear1(inputs))
        x = F.relu(self.linear2(x))
        out = self.linear_out(x)

        return out


class EncodeImputationDataset(Dataset):
    """Encode dataset."""

    def __init__(self, cells, assays, n_positions, chromosome, data_loc, verbose):
        self.cells = cells
        self.assays = assays
        self.n_cells = len(cells)
        self.n_assays = len(assays)
        self.n_positions = n_positions
        self.chromosome = chromosome
        self.data = torch.empty((self.n_cells, self.n_assays, self.n_positions_25bp), dtype=torch.float32)

        cell_assay_indexes = list()
        for i, cell in enumerate(self.cells):
            for j, assay in enumerate(self.assays):
                filename = os.path.join(data_loc, "{}{}.npz".format(cell, assay))
                if os.path.exists(filename):
                    if verbose:
                        print("load data {}...".format(filename))

                    with np.load(filename) as data:
                        self.data[i, j, :] = torch.as_tensor(data[self.chromosome])

                    cell_assay_indexes.append((i, j))

        self.cell_assay_indexes = torch.as_tensor(np.array(cell_assay_indexes))
        self.length = len(cell_assay_indexes) * n_positions_25bp

    def __len__(self):
        return self.n_positions

    def __getitem__(self, index):
        cell_assay_index = index // self.n_positions_25bp
        genomic_25bp_index = index - cell_assay_index * self.n_positions_25bp
        genomic_250bp_index = genomic_25bp_index // 10
        genomic_5kbp_index = genomic_25bp_index // 200

        cell_index, assay_index = self.cell_assay_indexes[cell_assay_index]
        x = np.array([cell_index, assay_index, genomic_25bp_index, genomic_250bp_index, genomic_5kbp_index])

        return torch.as_tensor(x), self.data[cell_index][assay_index][genomic_25bp_index]


def plot_history(train_loss, valid_loss, chrom):
    plt.subplot(1, 2, 1)
    plt.plot(train_loss)
    plt.title("Training loss", fontweight='bold')

    plt.subplot(1, 2, 2)
    plt.plot(valid_loss)
    plt.title("Validation loss", fontweight='bold')

    output_filename = os.path.join(vis_loc, "avocado.{}.pdf".format(chrom))
    plt.tight_layout()
    plt.savefig(output_filename)


def main():
    args = parse_args()

    assert args.chromosome is not None, "please choose a chromosome..."
    assert args.resolution in [0, 1, 2], "please choose a valid resolution..."

    seed_torch(seed=args.seed)

    cells, assays = get_cells_assays()

    chrom_size = chrom_size_dict[args.chrom]

    if args.resolution == 0:
        n_positions = chrom_size
    elif args.resolution == 1:
        n_positions = chrom_size // 10
    elif args.resolution == 1:
        n_positions = chrom_size // 200

    model_path = os.path.join(model_loc, "_{}.pth".format(args.chrom))

    embedding = GenomicPositionEmbedding(n_positions=n_positions, n_output_units=n_output_units)

    embedding.cuda()

    print("loading data...")
    train_dataset = EncodeImputationDataset(cells=cells, assays=assays, n_positions_25bp=n_positions_25bp,
                                            chromosome=args.chrom, data_loc=training_data_loc,
                                            verbose=args.verbose)

    print("creating dataloader...")
    train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, pin_memory=True,
                                  batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True)

    valid_dataloader = DataLoader(dataset=valid_dataset, shuffle=False, pin_memory=True,
                                  batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False)

    print('training data: %d' % (len(train_dataloader)))
    print('validation data: %d' % (len(valid_dataloader)))

    criterion = nn.MSELoss()

    history_train_loss = list()
    history_valid_loss = list()

    # initial loss
    ini_train_loss = 0
    ini_valid_loss = 0

    start = time.time()

    avocado.eval()
    for x, y in train_dataloader:
        y_pred = avocado(x.cuda()).reshape(-1)
        ini_train_loss += criterion(y.cuda(), y_pred).item()

    for x, y in valid_dataloader:
        y_pred = avocado(x.cuda()).reshape(-1)
        ini_valid_loss += criterion(y.cuda(), y_pred).item()

    ini_train_loss /= len(train_dataloader)
    ini_valid_loss /= len(valid_dataloader)

    history_train_loss.append(ini_train_loss)
    history_valid_loss.append(ini_valid_loss)

    secs = time.time() - start
    m, s = divmod(secs, 60)
    h, m = divmod(m, 60)

    print('epoch: %d, training loss: %.3f, validation loss: %.3f, time: %dh %dm %ds' % (0, ini_train_loss,
                                                                                        ini_valid_loss, h, m, s))

    best_valid_loss = ini_valid_loss
    optimizer = optimizers.Adam(avocado.parameters())
    for epoch in range(args.epochs):
        # training
        avocado.train()
        train_loss = 0.0

        start = time.time()

        for x, y in train_dataloader:
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            y_pred = avocado(x).reshape(-1)
            loss = criterion(y, y_pred)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # validation
        avocado.eval()
        valid_loss = 0.0
        for x, y in valid_dataloader:
            y_pred = avocado(x.cuda()).reshape(-1)
            valid_loss += criterion(y.cuda(), y_pred).item()

        train_loss /= len(train_dataloader)
        valid_loss /= len(valid_dataloader)

        secs = time.time() - start
        m, s = divmod(secs, 60)
        h, m = divmod(m, 60)

        print('epoch: %d, training loss: %.3f, validation loss: %.3f, time: %dh %dm %ds' % (epoch + 1, train_loss,
                                                                                            valid_loss, h, m, s))

        history_train_loss.append(train_loss)
        history_valid_loss.append(valid_loss)

        plot_history(history_train_loss, history_valid_loss, args.chrom)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(avocado.state_dict(), model_path)


if __name__ == '__main__':
    main()
