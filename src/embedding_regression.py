import os
import argparse
import numpy as np
import random
import time
import warnings
import pathlib
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.optim as optimizers
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset

training_data_tsv = "/hpcwork/izkf/projects/ENCODEImputation/local/TSV/metadata_training_data.tsv"
validation_data_tsv = "/hpcwork/izkf/projects/ENCODEImputation/local/TSV/metadata_validation_data.tsv"

training_data_loc = "/hpcwork/izkf/projects/ENCODEImputation/local/NPYFilesArcSinh/training_data"
validation_data_loc = "/hpcwork/izkf/projects/ENCODEImputation/local/NPYFilesArcSinh/validation_data"

model_loc = "/hpcwork/izkf/projects/ENCODEImputation/exp/Li/Models/EmbeddingRegression"
vis_loc = "/home/rs619065/EncodeImputation/vis/EmbeddingRegression"
history_loc = "/home/rs619065/EncodeImputation/history/EmbeddingRegression"

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
    parser.add_argument("-c", "--chrom", type=str, default=None)
    parser.add_argument("-bs", "--batch_size", type=int, default=20480)
    parser.add_argument("-e", "--epochs", type=int, default=30)
    parser.add_argument("-s", "--seed", type=int, default=2019)
    parser.add_argument("-v", "--verbose", type=int, default=0)
    parser.add_argument("-w", "--num_workers", type=int, default=24)

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


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class EmbeddingRegression(nn.Module):
    def __init__(self, n_cells, n_assays, n_positions_25bp, n_positions_250bp, n_positions_5kbp,
                 cell_embedding_dim=10, assay_embedding_dim=10, positions_25bp_embedding_dim=25,
                 positions_250bp_embedding_dim=25, positions_5kbp_embedding_dim=25, n_hidden_units=256):
        super(EmbeddingRegression, self).__init__()

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
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        cell_index = x[:, 0]
        assay_index = x[:, 1]
        positions_25bp_index = x[:, 2]
        positions_250bp_index = x[:, 3]
        positions_5kbp_index = x[:, 4]

        inputs = torch.cat((self.cell_embedding.weight[cell_index],
                            self.assay_embedding.weight[assay_index],
                            self.positions_25bp_embedding.weight[positions_25bp_index],
                            self.positions_250bp_embedding.weight[positions_250bp_index],
                            self.positions_5kbp_embedding.weight[positions_5kbp_index]), 1)

        x = F.relu(self.linear1(inputs))
        x = self.dropout(x)
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
        self.data = torch.empty((self.n_cells, self.n_assays, self.n_positions_25bp), dtype=torch.float32)

        cell_assay_indexes = list()
        for i, cell in enumerate(self.cells):
            for j, assay in enumerate(self.assays):
                filename = os.path.join(data_loc, "{}{}.npy".format(cell, assay))
                if os.path.exists(filename):
                    print("load data {}...".format(filename))
                    data = np.load(filename, allow_pickle=True)[()]
                    self.data[i, j, :] = torch.as_tensor(data[self.chromosome])
                    cell_assay_indexes.append((i, j))

        self.cell_assay_indexes = torch.as_tensor(np.array(cell_assay_indexes))
        self.length = len(cell_assay_indexes) * n_positions_25bp

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        cell_assay_index = index // self.n_positions_25bp
        genomic_25bp_index = index - cell_assay_index * self.n_positions_25bp
        genomic_250bp_index = genomic_25bp_index // 10
        genomic_5kbp_index = genomic_25bp_index // 200

        cell_index, assay_index = self.cell_assay_indexes[cell_assay_index]
        x = np.array([cell_index, assay_index, genomic_25bp_index, genomic_250bp_index, genomic_5kbp_index])

        return torch.as_tensor(x), self.data[cell_index][assay_index][genomic_25bp_index]


class ReduceLROnPlateau(object):
    """Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This scheduler reads a metrics
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    Args:
        factor: factor by which the learning rate will
            be reduced. new_lr = lr * factor
        patience: number of epochs with no improvement
            after which learning rate will be reduced.
        verbose: int. 0: quiet, 1: update messages.
        mode: one of {min, max}. In `min` mode,
            lr will be reduced when the quantity
            monitored has stopped decreasing; in `max`
            mode it will be reduced when the quantity
            monitored has stopped increasing.
        epsilon: threshold for measuring the new optimum,
            to only focus on significant changes.
        cooldown: number of epochs to wait before resuming
            normal operation after lr has been reduced.
        min_lr: lower bound on the learning rate.
    """

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 verbose=0, epsilon=1e-4, cooldown=0, min_lr=0.0):
        super(ReduceLROnPlateau, self).__init__()

        if factor >= 1.0:
            raise ValueError('ReduceLROnPlateau '
                             'does not support a factor >= 1.0.')
        self.factor = factor
        self.min_lr = min_lr
        self.epsilon = epsilon
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.monitor_op = None
        self.wait = 0
        self.best = 0
        self.mode = mode
        assert isinstance(optimizer, Optimizer)
        self.optimizer = optimizer
        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter.
        """
        if self.mode not in ['min', 'max']:
            raise RuntimeError('Learning Rate Plateau Reducing mode %s is unknown!')
        if self.mode == 'min':
            self.monitor_op = lambda a, b: np.less(a, b - self.epsilon)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.epsilon)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0
        self.lr_epsilon = self.min_lr * 1e-4

    def reset(self):
        self._reset()

    def step(self, metrics, epoch):
        current = metrics
        if current is None:
            warnings.warn('Learning Rate Plateau Reducing requires metrics available!', RuntimeWarning)
        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                if self.wait >= self.patience:
                    for param_group in self.optimizer.param_groups:
                        old_lr = float(param_group['lr'])
                        if old_lr > self.min_lr + self.lr_epsilon:
                            new_lr = old_lr * self.factor
                            new_lr = max(new_lr, self.min_lr)
                            param_group['lr'] = new_lr
                            if self.verbose > 0:
                                print('\nEpoch %05d: reducing learning rate to %s.' % (epoch, new_lr))
                            self.cooldown_counter = self.cooldown
                            self.wait = 0
                self.wait += 1

    def in_cooldown(self):
        return self.cooldown_counter > 0


def plot_history(train_loss, valid_loss, chrom):
    plt.subplot(1, 2, 1)
    plt.plot(train_loss)
    plt.title("Training loss", fontweight='bold')

    plt.subplot(1, 2, 2)
    plt.plot(valid_loss)
    plt.title("Validation loss", fontweight='bold')

    output_filename = os.path.join(vis_loc, "{}.pdf".format(chrom))
    plt.tight_layout()
    plt.savefig(output_filename)


def write_history(train_loss, valid_loss, chrom):
    output_filename = os.path.join(history_loc, "{}.txt".format(chrom))

    with open(output_filename, "w") as f:
        f.write("TrainLoss" + "\t" + "ValidLoss" + "\n")
        for i, loss in enumerate(train_loss):
            f.write(str(train_loss[i]) + "\t" + str(valid_loss[i]) + "\n")


def main():
    args = parse_args()

    assert args.chrom is not None, "please choose a chromosome..."

    seed_torch(seed=args.seed)

    cells, assays = get_cells_assays()

    n_positions_25bp = chrom_size_dict[args.chrom]

    n_positions_250bp, n_positions_5kbp = n_positions_25bp // 10 + 1, n_positions_25bp // 200 + 1

    model_path = os.path.join(model_loc, "{}.pth".format(args.chrom))

    embedding_regression = EmbeddingRegression(n_cells=len(cells),
                                               n_assays=len(assays),
                                               n_positions_25bp=n_positions_25bp,
                                               n_positions_250bp=n_positions_250bp,
                                               n_positions_5kbp=n_positions_5kbp)

    if os.path.exists(model_path):
        embedding_regression.load_state_dict(torch.load(model_path))

    if torch.cuda.is_available():
        embedding_regression.cuda()

    pathlib.Path(model_loc).mkdir(parents=True, exist_ok=True)
    pathlib.Path(vis_loc).mkdir(parents=True, exist_ok=True)
    pathlib.Path(history_loc).mkdir(parents=True, exist_ok=True)

    print("loading data...")
    train_dataset = EncodeImputationDataset(cells=cells, assays=assays, n_positions_25bp=n_positions_25bp,
                                            chromosome=args.chrom, data_loc=training_data_loc)

    valid_dataset = EncodeImputationDataset(cells=cells, assays=assays, n_positions_25bp=n_positions_25bp,
                                            chromosome=args.chrom, data_loc=validation_data_loc)

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

    embedding_regression.eval()
    if torch.cuda.is_available():
        for x, y in train_dataloader:
            y_pred = embedding_regression(x.cuda()).reshape(-1)
            ini_train_loss += criterion(y.cuda(), y_pred).item()

        for x, y in valid_dataloader:
            y_pred = embedding_regression(x.cuda()).reshape(-1)
            ini_valid_loss += criterion(y.cuda(), y_pred).item()
    else:
        for x, y in train_dataloader:
            y_pred = embedding_regression(x).reshape(-1)
            ini_train_loss += criterion(y, y_pred).item()

        for x, y in valid_dataloader:
            y_pred = embedding_regression(x).reshape(-1)
            ini_valid_loss += criterion(y, y_pred).item()

    ini_train_loss /= len(train_dataloader)
    ini_valid_loss /= len(valid_dataloader)

    history_train_loss.append(ini_train_loss)
    history_valid_loss.append(ini_valid_loss)

    secs = time.time() - start
    m, s = divmod(secs, 60)
    h, m = divmod(m, 60)

    print('epoch: %d, training loss: %.8f, validation loss: %.8f, time: %dh %dm %ds' % (0, ini_train_loss,
                                                                                        ini_valid_loss, h, m, s))
    optimizer = optimizers.Adam(embedding_regression.parameters())
    for epoch in range(args.epochs):
        # training
        embedding_regression.train()
        train_loss = 0.0

        start = time.time()

        if torch.cuda.is_available():
            for x, y in train_dataloader:
                x, y = x.cuda(), y.cuda()
                optimizer.zero_grad()
                y_pred = embedding_regression(x).reshape(-1)
                loss = criterion(y, y_pred)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
        else:
            for x, y in train_dataloader:
                optimizer.zero_grad()
                y_pred = embedding_regression(x).reshape(-1)
                loss = criterion(y, y_pred)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

        # validation
        embedding_regression.eval()
        valid_loss = 0.0
        if torch.cuda.is_available():
            for x, y in valid_dataloader:
                y_pred = embedding_regression(x.cuda()).reshape(-1)
                valid_loss += criterion(y.cuda(), y_pred).item()
        else:
            for x, y in valid_dataloader:
                y_pred = embedding_regression(x).reshape(-1)
                valid_loss += criterion(y, y_pred).item()

        train_loss /= len(train_dataloader)
        valid_loss /= len(valid_dataloader)

        secs = time.time() - start
        m, s = divmod(secs, 60)
        h, m = divmod(m, 60)

        print('epoch: %d, training loss: %.8f, validation loss: %.8f, time: %dh %dm %ds' % (epoch + 1, train_loss,
                                                                                            valid_loss, h, m, s))
        history_train_loss.append(train_loss)
        history_valid_loss.append(valid_loss)

        plot_history(history_train_loss, history_valid_loss, args.chrom)
        write_history(history_train_loss, history_valid_loss, args.chrom)

        torch.save(embedding_regression.state_dict(), model_path)


if __name__ == '__main__':
    main()
