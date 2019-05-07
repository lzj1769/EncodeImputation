import os
import argparse
import numpy as np
from tqdm import tqdm
import random
import time

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.optim as optimizers
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

training_data_csv = "/hpcwork/izkf/projects/ENCODEImputation/data/training_data/metadata_training_data.tsv"
training_data_loc = "/hpcwork/izkf/projects/ENCODEImputation/input/training_data"

validation_data_csv = "/hpcwork/izkf/projects/ENCODEImputation/data/validation_data/metadata_training_data.tsv"
validation_data_loc = "/hpcwork/izkf/projects/ENCODEImputation/input/validation_data"

model_loc = "/hpcwork/izkf/projects/ENCODEImputation/model"
vis_loc = "/home/rs619065/EncodeImputation/vis"

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
    parser.add_argument("-bs", "--batch_size", type=int, default=20480)
    parser.add_argument("-e", "--epochs", type=int, default=30)
    parser.add_argument("-s", "--seed", type=int, default=2019)
    parser.add_argument("-v", "--verbose", type=int, default=0)
    parser.add_argument("-w", "--num_workers", type=int, default=24)

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

    def __init__(self, cells, assays, n_positions_25bp, chromosome, data_loc, verbose):
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
        return self.length

    def __getitem__(self, index):
        cell_assay_index = index // self.n_positions_25bp
        genomic_25bp_index = index - cell_assay_index * self.n_positions_25bp
        genomic_250bp_index = genomic_25bp_index // 10
        genomic_5kbp_index = genomic_25bp_index // 200

        cell_index, assay_index = self.cell_assay_indexes[cell_assay_index]
        x = np.array([cell_index, assay_index, genomic_25bp_index, genomic_250bp_index, genomic_5kbp_index])

        return torch.as_tensor(x), self.data[cell_index][assay_index][genomic_25bp_index]


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
    assert embedding in ['cell_embedding', 'assay_embedding', 'positions_25bp_embedding',
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

    assert args.chrom is not None, "please choose a chromosome..."

    seed_torch(seed=args.seed)

    cells, assays = get_cells_assays()

    chrom_size = chrom_size_dict[args.chrom]

    n_positions_25bp, n_positions_250bp, n_positions_5kbp = get_positions(chrom_size)

    model_path = os.path.join(model_loc, "avocado_{}.pth".format(args.chrom))

    avocado = Avocado(n_cells=len(cells),
                      n_assays=len(assays),
                      n_positions_25bp=n_positions_25bp,
                      n_positions_250bp=n_positions_250bp,
                      n_positions_5kbp=n_positions_5kbp)

    if os.path.exists(model_path):
        avocado.load_state_dict(torch.load(model_path))

    # use multi-GPUs
    # avocado = nn.DataParallel(avocado)
    avocado.cuda()

    print("loading data...")
    train_dataset = EncodeImputationDataset(cells=cells, assays=assays, n_positions_25bp=n_positions_25bp,
                                            chromosome=args.chrom, data_loc=training_data_loc,
                                            verbose=args.verbose)

    valid_dataset = EncodeImputationDataset(cells=cells, assays=assays, n_positions_25bp=n_positions_25bp,
                                            chromosome=args.chrom, data_loc=validation_data_loc,
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
    if args.verbose:
        for x, y in tqdm(train_dataloader):
            y_pred = avocado(x.cuda()).reshape(-1)
            ini_train_loss += criterion(y.cuda(), y_pred).item()

        for x, y in tqdm(valid_dataloader):
            y_pred = avocado(x.cuda()).reshape(-1)
            ini_valid_loss += criterion(y.cuda(), y_pred).item()
    else:
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
    for epoch in tqdm(range(args.epochs)):
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
