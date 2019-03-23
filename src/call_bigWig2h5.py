from __future__ import print_function

import os
from configure import *

for chrom in chrom_list[:1]:
    job_name = chrom
    command = "sbatch -J " + job_name + " -o " + "./cluster_out/" + job_name + "_out.txt -e " + \
              "./cluster_err/" + job_name + "_err.txt -t 120:00:00 --mem 100G "
    command += "./bigWig2h5.zsh "
    os.system(command + " " + chrom)
