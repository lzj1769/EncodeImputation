import os
import argparse
import numpy as np
import random
import time
import pathlib
import torch
from torch import nn
import torch.optim as optimizers
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

training_data_loc = "/hpcwork/izkf/projects/ENCODEImputation/local/NPYFilesFinal/training_data"
validation_data_loc = "/hpcwork/izkf/projects/ENCODEImputation/local/NPYFilesFinal/validation_data"

embedding_loc = "/hpcwork/izkf/projects/ENCODEImputation/exp/Li/Models/EmbeddingRegression"
model_loc = "/hpcwork/izkf/projects/ENCODEImputation/exp/Li/Models/EncodeImputationNet"

vis_loc = "/home/rs619065/EncodeImputation/vis/EncodeImputationNet"
history_loc = "/home/rs619065/EncodeImputation/history/EncodeImputationNet"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-chr", "--chrom", type=str, default=None)
    parser.add_argument("-bs", "--batch_size", type=int, default=20480)
    parser.add_argument("-e", "--epochs", type=int, default=30)
    parser.add_argument("-s", "--seed", type=int, default=2019)
    parser.add_argument("-v", "--verbose", type=int, default=0)
    parser.add_argument("-w", "--num_workers", type=int, default=8)

    return parser.parse_args()


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class EncodeImputationNet(nn.Module):
    def __init__(self, cell_embedding, assay_embedding, positions_25bp_embedding,
                 positions_250bp_embedding, positions_5kbp_embedding, n_hidden_units=256):
        super(EncodeImputationNet, self).__init__()
        n_cells, cell_embedding_dim = cell_embedding.shape
        n_assays, assay_embedding_dim = assay_embedding.shape
        n_positions_25bp, positions_25bp_embedding_dim = positions_25bp_embedding.shape
        n_positions_250bp, positions_250bp_embedding_dim = positions_250bp_embedding.shape
        n_positions_5kbp, positions_5kbp_embedding_dim = positions_5kbp_embedding.shape

        # cell embedding matrix
        self.cell_embedding = nn.Embedding(num_embeddings=n_cells, embedding_dim=cell_embedding_dim)
        self.assay_embedding = nn.Embedding(num_embeddings=n_assays, embedding_dim=assay_embedding_dim)
        self.positions_25bp_embedding = nn.Embedding(num_embeddings=n_positions_25bp,
                                                     embedding_dim=positions_25bp_embedding_dim)
        self.positions_250bp_embedding = nn.Embedding(num_embeddings=n_positions_250bp,
                                                      embedding_dim=positions_250bp_embedding_dim)
        self.positions_5kbp_embedding = nn.Embedding(num_embeddings=n_positions_5kbp,
                                                     embedding_dim=positions_5kbp_embedding_dim)

        self.cell_embedding.load_state_dict({'weight': cell_embedding})
        self.assay_embedding.load_state_dict({'weight': assay_embedding})
        self.positions_25bp_embedding.load_state_dict({'weight': positions_25bp_embedding})
        self.positions_250bp_embedding.load_state_dict({'weight': positions_250bp_embedding})
        self.positions_5kbp_embedding.load_state_dict({'weight': positions_5kbp_embedding})

        self.cell_embedding.weight.requires_grad = False
        self.assay_embedding.weight.requires_grad = False
        self.positions_25bp_embedding.weight.requires_grad = False
        self.positions_250bp_embedding.weight.requires_grad = False
        self.positions_5kbp_embedding.weight.requires_grad = False

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

    def __init__(self, x, y):
        self.x = torch.as_tensor(x)
        self.y = torch.as_tensor(y)
        self.length = len(y)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def write_history(train_loss, valid_loss, chrom):
    output_filename = os.path.join(history_loc, "{}.txt".format(chrom))

    if os.path.exists(output_filename):
        with open(output_filename, "a") as f:
            f.write(str(train_loss) + "\t" + str(valid_loss) + "\n")
    else:
        with open(output_filename, "w") as f:
            f.write("TrainLoss" + "\t" + "ValidLoss" + "\n")
            f.write(str(train_loss) + "\t" + str(valid_loss) + "\n")


def main():
    args = parse_args()

    assert args.chrom is not None, "please choose a chromosome..."

    seed_torch(seed=args.seed)

    model_path = os.path.join(model_loc, "{}.pth".format(args.chrom))
    embedding_path = os.path.join(embedding_loc, "{}.pth".format(args.chrom))
    embedding = torch.load(embedding_path, map_location='cpu')
    cell_embedding = embedding['cell_embedding.weight']
    assay_embedding = embedding['assay_embedding.weight']
    positions_25bp_embedding = embedding['positions_25bp_embedding.weight']
    positions_250bp_embedding = embedding['positions_250bp_embedding.weight']
    positions_5kbp_embedding = embedding['positions_5kbp_embedding.weight']

    encode_net = EncodeImputationNet(cell_embedding=cell_embedding,
                                     assay_embedding=assay_embedding,
                                     positions_25bp_embedding=positions_25bp_embedding,
                                     positions_250bp_embedding=positions_250bp_embedding,
                                     positions_5kbp_embedding=positions_5kbp_embedding)

    if os.path.exists(model_path):
        encode_net.load_state_dict(torch.load(model_path, map_location='cpu'))

    encode_net.cuda()

    pathlib.Path(model_loc).mkdir(parents=True, exist_ok=True)
    pathlib.Path(vis_loc).mkdir(parents=True, exist_ok=True)
    pathlib.Path(history_loc).mkdir(parents=True, exist_ok=True)

    print("loading data...")
    train_x = np.load(os.path.join(training_data_loc, "{}_x.npy".format(args.chrom)))
    train_y = np.load(os.path.join(training_data_loc, "{}_y.npy".format(args.chrom)))
    valid_x = np.load(os.path.join(validation_data_loc, "{}_x.npy".format(args.chrom)))
    valid_y = np.load(os.path.join(validation_data_loc, "{}_y.npy".format(args.chrom)))

    train_dataset = EncodeImputationDataset(x=train_x, y=train_y)
    valid_dataset = EncodeImputationDataset(x=valid_x, y=valid_y)

    print("creating dataloader...")
    train_dataloader = DataLoader(dataset=train_dataset, shuffle=False, pin_memory=True,
                                  batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False)

    valid_dataloader = DataLoader(dataset=valid_dataset, shuffle=False, pin_memory=True,
                                  batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False)

    print('training data: %d' % (len(train_dataloader)))
    print('validation data: %d' % (len(valid_dataloader)))

    criterion = nn.MSELoss()
    optimizer = optimizers.Adam(encode_net.parameters())
    for epoch in range(args.epochs):
        # training
        encode_net.train()
        train_loss = 0.0

        start = time.time()
        for x, y in train_dataloader:
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            y_pred = encode_net(x).reshape(-1)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # validation
        encode_net.eval()
        valid_loss = 0.0
        for x, y in valid_dataloader:
            y_pred = encode_net(x.cuda()).reshape(-1)
            valid_loss += criterion(y_pred, y.cuda()).item()

        train_loss /= len(train_dataloader)
        valid_loss /= len(valid_dataloader)

        secs = time.time() - start
        m, s = divmod(secs, 60)
        h, m = divmod(m, 60)

        print('epoch: %d, training loss: %.8f, validation loss: %.8f, time: %dh %dm %ds' % (epoch + 1, train_loss,
                                                                                            valid_loss, h, m, s))

        write_history(train_loss, valid_loss, args.chrom)
        torch.save(encode_net.state_dict(), model_path)


if __name__ == '__main__':
    main()
