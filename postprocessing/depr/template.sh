#!/bin/bash
#PBS -N ctsm6_key
#PBS -q casper
#PBS -l walltime=1:00:00
#PBS -A P93300041
#PBS -j oe
#PBS -k eod
#PBS -l select=1:ncpus=1

source ~/.bashrc
conda activate mlenv

python postp_lhc_pxbmean.py key
