from configure import *


def get_train_cell_assays():
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


def get_validation_cell_assays():
    cell_type = []
    assays = []

    f = open(validation_data_csv)
    f.readline()

    for line in f.readlines():
        ll = line.strip().split("\t")
        cell_type.append(ll[1])
        assays.append(ll[4])

    f.close()

    return list(set(cell_type)), list(set(assays))
