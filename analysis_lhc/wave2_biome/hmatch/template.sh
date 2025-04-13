#!/bin/bash
#PBS -N wave2_hmatch_key
#PBS -q casper
#PBS -l walltime=3:00:00
#PBS -A P93300041
#PBS -j oe
#PBS -k eod
#PBS -l select=1:ncpus=1

source ~/.bashrc
conda activate mlenv

python sample_hmatch_save.py key

