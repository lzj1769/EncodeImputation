from __future__ import print_function

import os

validation_data_tsv = "/hpcwork/izkf/projects/ENCODEImputation/local/TSV/metadata_validation_data.tsv"
output_loc = "/work/rwth0233/ENCODEImputation/Prediction/EmbeddingRegression/validation/NPY"
cells = []
assays = []
f = open(validation_data_tsv)
f.readline()
for line in f.readlines():
    ll = line.strip().split("\t")
    cells.append(ll[1])
    assays.append(ll[2])
f.close()

for i, cell in enumerate(cells):
    assay = assays[i]
    job_name = "pred_{}{}".format(cell, assay)
    command = "sbatch -J " + job_name + " -o " + "./cluster_out/" + job_name + "_out.txt -e " + \
              "./cluster_err/" + job_name + "_err.txt -t 10:00:00 --mem 180G "
    command += "-c 12 -A rwth0233 regression_predict.zsh"
    os.system(command + " " + cell + " " + assay + " " + output_loc)
