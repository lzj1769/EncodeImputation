import os
import numpy as np
import torch
import pandas as pd

training_data_tsv = "/hpcwork/izkf/projects/ENCODEImputation/local/TSV/metadata_training_data.tsv"
validation_data_tsv = "/hpcwork/izkf/projects/ENCODEImputation/local/TSV/metadata_validation_data.tsv"

model_loc = "/hpcwork/izkf/projects/ENCODEImputation/exp/Li/Models/EmbeddingRegression"
output_loc = "/home/rs619065/EncodeImputation/EmbeddingMatrix"


def get_cells_assays():
    cells = []
    assays = []

    f = open(training_data_tsv)
    f.readline()

    for line in f.readlines():
        ll = line.strip().split("\t")
        if ll[3] not in cells:
            cells.append(ll[3])

        if ll[4] not in assays:
            assays.append(ll[4])
    f.close()

    return cells, assays


def main():
    cells, assays = get_cells_assays()
    chroms = ['chr' + str(i) for i in range(1, 23)] + ['chrX']

    cells_embedding = np.empty(shape=(len(cells), 0))
    assays_embedding = np.empty(shape=(len(assays), 0))

    for chrom in chroms:
        model = torch.load(os.path.join(model_loc, "{}.pth".format(chrom)), map_location="cpu")
        cells_embedding = np.append(cells_embedding, model['cell_embedding.weight'], axis=1)
        assays_embedding = np.append(assays_embedding, model['assay_embedding.weight'], axis=1)

    df = pd.DataFrame(data=np.transpose(cells_embedding), columns=cells)
    filename = os.path.join(output_loc, "Cells.txt")
    df.to_csv(filename, sep="\t", index=False)

    df = pd.DataFrame(data=np.transpose(assays_embedding), columns=assays)
    filename = os.path.join(output_loc, "Assays.txt")
    df.to_csv(filename, sep="\t", index=False)


if __name__ == '__main__':
    main()
