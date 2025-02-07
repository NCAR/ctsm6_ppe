#!/bin/bash
#PBS -N pft_hmatch_num
#PBS -q casper
#PBS -l walltime=1:00:00
#PBS -A P93300041
#PBS -j oe
#PBS -k eod
#PBS -l select=1:ncpus=1

source ~/.bashrc
conda activate /glade/work/linnia/conda-envs/mlenv

python pft_hmatch.py num

