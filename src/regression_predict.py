import os
import argparse
import numpy as np
import random
import time
import pathlib

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

training_data_tsv = "/hpcwork/izkf/projects/ENCODEImputation/local/TSV/metadata_training_data.tsv"
validation_data_tsv = "/hpcwork/izkf/projects/ENCODEImputation/local/TSV/metadata_validation_data.tsv"

model_loc = "/home/rs619065/EncodeImputation/EmbeddingRegression"
training_prediction_loc = "/work/rwth0233/ENCODEImputation/Prediction/EmbeddingRegression/training"
validation_prediction_loc = "/work/rwth0233/ENCODEImputation/Prediction/EmbeddingRegression/validation"

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
    parser.add_argument("-c", "--cell", type=str, default=None)
    parser.add_argument("-a", "--assay", type=str, default=None)
    parser.add_argument("-bs", "--batch_size", type=int, default=40000)
    parser.add_argument("-s", "--seed", type=int, default=2019)
    parser.add_argument("-v", "--verbose", type=int, default=0)
    parser.add_argument("-w", "--num_workers", type=int, default=12)

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

    def __init__(self, cell_index, assay_index, n_positions_25bp):
        self.cell_index = cell_index
        self.assay_index = assay_index
        self.n_positions_25bp = n_positions_25bp

    def __len__(self):
        return self.n_positions_25bp

    def __getitem__(self, index):
        genomic_250bp_index = index // 10
        genomic_5kbp_index = index // 200

        x = np.array([self.cell_index, self.assay_index, index, genomic_250bp_index, genomic_5kbp_index])

        return torch.as_tensor(x)


def main():
    args = parse_args()

    if args.chrom is None:
        args.chrom = chrom_size_dict.keys()

    seed_torch(seed=args.seed)

    cells, assays = get_cells_assays()

    cell_index = cells.index(args.cell)
    assay_index = assays.index(args.assay)

    n_positions_25bp = chrom_size_dict[args.chrom]

    n_positions_250bp, n_positions_5kbp = n_positions_25bp // 10 + 1, n_positions_25bp // 200 + 1

    model_path = os.path.join(model_loc, "{}.pth".format(args.chrom))

    embedding_regression = EmbeddingRegression(n_cells=len(cells),
                                               n_assays=len(assays),
                                               n_positions_25bp=n_positions_25bp,
                                               n_positions_250bp=n_positions_250bp,
                                               n_positions_5kbp=n_positions_5kbp)

    if torch.cuda.is_available():
        embedding_regression.load_state_dict(torch.load(model_path))
    else:
        embedding_regression.load_state_dict(torch.load(model_path, map_location='cpu'))

    pathlib.Path(model_loc).mkdir(parents=True, exist_ok=True)
    dataset = EncodeImputationDataset(cell_index=cell_index, assay_index=assay_index, n_positions_25bp=n_positions_25bp)
    dataloader = DataLoader(dataset=dataset, shuffle=False, pin_memory=True,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers, drop_last=False)

    start = time.time()

    embedding_regression.eval()
    pred = np.empty(0)

    if torch.cuda.is_available():
        for x in dataloader:
            y_pred = embedding_regression(x.cuda()).reshape(-1)
            pred = np.append(pred, y_pred.detach().numpy())
    else:
        for x in dataloader:
            y_pred = embedding_regression(x).reshape(-1)
            pred = np.append(pred, y_pred.detach().numpy())

    filename = os.path.join(training_prediction_loc, "{}{}_{}".format(args.cell, args.assay, args.chrom))
    np.save(filename, np.sinh(pred))
    secs = time.time() - start
    m, s = divmod(secs, 60)
    h, m = divmod(m, 60)

    print('time: %dh %dm %ds' % (h, m, s))


if __name__ == '__main__':
    main()
