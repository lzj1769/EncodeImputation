from __future__ import print_function

import os
from utils import *

for chrom in chrom_list:
    if chrom not in ['chr21', 'chr22']:
        continue

    job_name = "avocado_{}".format(chrom)
    command = "sbatch -J " + job_name + " -o " + "./cluster_out/" + job_name + "_out.txt -e " + \
              "./cluster_err/" + job_name + "_err.txt -t 120:00:00 --mem 100G -c 24 --partition=c18g --gres=gpu:1 "
    command += "./avocado.zsh "
    os.system(command + " " + chrom)
