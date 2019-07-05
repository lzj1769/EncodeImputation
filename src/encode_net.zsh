#!/bin/bash

################################################################
# PATH
export PATH=/usr/bin:$PATH
export PATH=/usr/local_host/bin:$PATH
export PATH=/home/rs619065/local/bin:$PATH
export PATH=/home/rs619065/.local/bin:$PATH
export PATH=/home/rs619065/local/bamtools/bin:$PATH
export PATH=/home/rs619065/perl5/bin:$PATH

################################################################
# LIBRARYPATH
export LD_LIBRARY_PATH=/home/rs619065/local/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=24

source ~/miniconda2/bin/activate py36env
python -m cProfile -o encode_net.txt encode_net.py -c $1 -e $2
