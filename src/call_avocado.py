from __future__ import print_function

import os
from utils import *

for chrom in chrom_list[:5]:
    job_name = chrom
    command = "sbatch -J " + job_name + " -o " + "./cluster_out/" + job_name + "_out.txt -e " + \
              "./cluster_err/" + job_name + "_err.txt -t 120:00:00 --mem 180G --partition=c18g --gres=gpu:1 "
    command += "./avocado.zsh "
    os.system(command + " " + chrom)
