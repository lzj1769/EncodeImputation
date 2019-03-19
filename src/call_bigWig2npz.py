from __future__ import print_function

import os

training_data = "/hpcwork/izkf/projects/ENCODEImputation/data/training_data/metadata_training_data.tsv"

f = open(training_data)
f.readline()

for line in f.readlines():
    ll = line.strip().split("\t")
    cell_type = ll[1]
    assay_index = ll[2]
    assay_type = ll[4]
    filename = ll[-1]

    input_filename = os.path.join("/hpcwork/izkf/projects/ENCODEImputation/data/training_data", filename)
    output_location = os.path.join("/hpcwork/izkf/projects/ENCODEImputation/input/training_data", assay_type)

    if not os.path.exists(output_location):
        os.system("mkdir -p " + output_location)

    job_name = "{}_{}".format(cell_type, assay_type)
    command = "sbatch -J " + job_name + " -o " + "./cluster_out/" + job_name + "_out.txt -e " + \
              "./cluster_err/" + job_name + "_err.txt -t 100:00:00 --mem 30G "
    command += " ./bigWig2npz.zsh "
    os.system(command + " " + input_filename + " " + output_location + " " + cell_type + " " + assay_type)
