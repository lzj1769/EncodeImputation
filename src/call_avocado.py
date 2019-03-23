from __future__ import print_function

import os
from utils import *

for chrom in chrom_list[18:]:
    job_name = "avocado_{}".format(chrom)
    command = "sbatch -J " + job_name + " -o " + "./cluster_out/" + job_name + "_out.txt -e " + \
              "./cluster_err/" + job_name + "_err.txt -t 120:00:00 --mem 100G -c 48 --partition=c18g --gres=gpu:2 "
    command += "./avocado.zsh "
    os.system(command + " " + chrom)
