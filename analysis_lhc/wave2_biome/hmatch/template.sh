#!/bin/bash
#PBS -N wave2_hmatch_num
#PBS -q casper
#PBS -l walltime=2:00:00
#PBS -A P93300041
#PBS -j oe
#PBS -k eod
#PBS -l select=1:ncpus=1

source ~/.bashrc
conda activate mlenv

python Ctree_hmatch.py num

