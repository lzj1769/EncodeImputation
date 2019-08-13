from __future__ import print_function

import os

# validation_data_tsv = "/hpcwork/izkf/projects/ENCODEImputation/local/TSV/metadata_validation_data.tsv"
# output_loc = "/hpcwork/izkf/projects/ENCODEImputation/exp/Li/Prediction/EmbeddingRegression/validation/NPY"
# cells = []
# assays = []
# f = open(validation_data_tsv)
# f.readline()
# for line in f.readlines():
#     ll = line.strip().split("\t")
#     cells.append(ll[1])
#     assays.append(ll[2])
# f.close()

# for i, cell in enumerate(cells):
#     assay = assays[i]
#     job_name = "pred_{}{}".format(cell, assay)
#     command = "sbatch -J " + job_name + " -o " + "./cluster_out/" + job_name + "_out.txt -e " + \
#               "./cluster_err/" + job_name + "_err.txt -t 10:00:00 --mem 180G "
#     command += "-c 12 -A rwth0233 regression_predict.zsh"
#     os.system(command + " " + cell + " " + assay + " " + output_loc)

output_loc = "/hpcwork/izkf/projects/ENCODEImputation/exp/Li/Prediction/EmbeddingRegression/test/NPY"
cell_assays_dict = dict()
cell_assays_dict['C05'] = ['M17', 'M18', 'M20', 'M29']
cell_assays_dict['C06'] = ['M16', 'M17', 'M28']
cell_assays_dict['C07'] = ['M20', 'M29']
cell_assays_dict['C12'] = ['M01', 'M02']
cell_assays_dict['C14'] = ['M01', 'M02', 'M16', 'M17', 'M22']
cell_assays_dict['C19'] = ['M16', 'M17', 'M18', 'M20', 'M22', 'M29']
cell_assays_dict['C22'] = ['M16', 'M17']
cell_assays_dict['C28'] = ['M17', 'M18', 'M22', 'M29']
cell_assays_dict['C31'] = ['M01', 'M02', 'M16', 'M29']
cell_assays_dict['C38'] = ['M01', 'M02', 'M17', 'M18', 'M20', 'M22', 'M29']
cell_assays_dict['C39'] = ['M16', 'M17', 'M18', 'M20', 'M22', 'M29']
cell_assays_dict['C40'] = ['M16', 'M17', 'M18', 'M20', 'M22', 'M29']
cell_assays_dict['C51'] = ['M16', 'M17', 'M18', 'M20', 'M29']


for cell in cell_assays_dict.keys():
    for assay in cell_assays_dict[cell]:
        job_name = "pred_{}{}".format(cell, assay)
        command = "sbatch -J " + job_name + " -o " + "./cluster_out/" + job_name + "_out.txt -e " + \
                  "./cluster_err/" + job_name + "_err.txt -t 10:00:00 --mem 180G "
        command += "-c 12 -A rwth0233 regression_predict.zsh"
        os.system(command + " " + cell + " " + assay + " " + output_loc)
