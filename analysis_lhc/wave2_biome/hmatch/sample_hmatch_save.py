import numpy as np
import pandas as pd
import xarray as xr
import os
import pickle
import gpflow

import sys
sys.path.append('/glade/u/home/linnia/ctsm6_ppe/')
from utils.pyfunctions import *
utils_path = '/glade/u/home/linnia/ctsm6_ppe/utils/'
from sampling_pyfunctions import *

import warnings
warnings.filterwarnings("ignore")

# input to script
key = sys.argv[1]

#############################################
# user mods
#############################################

outdir = '/glade/work/linnia/CLM6-PPE/ctsm6_wave1/NROY/'

#############################################
# Setup
#############################################
# Define biome settings: biome name, index, and PFTs 
biome_configs = [
    {'name': 'NaN','index':0, 'pfts':[np.NaN]},
    {'name': 'Tropical rainforest', 'index':1, 'pfts': [4]},
    {'name': 'Tropical savanna', 'index':2, 'pfts': [4,14]},
    {'name': 'Subtropical savanna', 'index':3, 'pfts': [4,6,14]},
    {'name': 'Broadleaf evergreen temperate tree', 'index':4, 'pfts':[5, 13, 14]},
    {'name': 'Grasslands', 'index': 5, 'pfts':[13,14]},
    {'name': 'Shrubland', 'index':6, 'pfts': [10,13,14]},
    {'name': 'Mixed deciduous temperate forest', 'index':7, 'pfts': [1, 7, 13, 14]},
    {'name': 'Conifer forest', 'index':8, 'pfts': [1, 2, 13, 14]},
    {'name': 'Siberian larch', 'index':9, 'pfts': [3,11,12]},
    {'name': 'Boreal forest', 'index':10, 'pfts': [2,11,12]},
    {'name': 'Broadleaf deciduous boreal trees', 'index':11, 'pfts': [2, 8, 12, 13]},
    {'name': 'Boreal shrubland', 'index': 12, 'pfts': [12]},
    {'name': 'Tundra', 'index': 13, 'pfts': [11, 12]},
]

# load observational data
obs = xr.open_dataset('../wave2_obsStatistics_sudokuBiomes.nc')

# info on parameter names
params_lhc = pd.read_csv('/glade/work/linnia/CLM6-PPE/ctsm6_lhc/ctsm6lhc_11262024.txt').drop(columns='member')

pft_params   = ['kmax','psi50','jmaxb0','slatop','lmr_intercept_atkin',
                'medlynslope','medlynintercept','froot_leaf','leafcn','leaf_long',
                'KCN','dleaf','r_mort','fsr_pft','xl']
pftix=np.array([p in pft_params for p in params_lhc.columns])
u_params = params_lhc.columns[~pftix]

pft_param_names = {i: [f"{param}_{i}" for param in pft_params] for i in range(1, 15)}

# Biome info
file='/glade/u/home/linnia/ctsm6_ppe/utils/sudoku_biomes.nc'
biomes=xr.open_dataset(file)
biome_names = biomes.biome_names.values

with open("/glade/u/home/linnia/ctsm6_ppe/utils/biome_pft_dict.pkl", "rb") as f:
    biome_pfts = pickle.load(f)

#############################################
# history matching 
#############################################

usamples = pd.read_csv('universal_samples_LHC100000.csv')

emulator_path = '/glade/u/home/linnia/ctsm6_ppe/analysis_lhc/wave2_biome/emulators_biome'

n_psamp = 10000
n_usets = 100
k = int(key)

param_sets = []
for u in range(int(k * n_usets), int((k + 1) * n_usets)):
    usample = usamples.iloc[[u]]
    psample = pd.DataFrame(np.random.rand(n_psamp,len(pft_params)),columns=pft_params)

    b_samples = calibration_tree(usample,psample,n_psamp,u_params,pft_param_names,emulator_path,obs,biome_configs)
    # if none of the samples are plausible for any given biome, continue to next universal set
    if len(b_samples)==1:
        continue
    else:
        s = create_master_sample(b_samples,pft_param_names)
    
        param_sets.append(s)

master_sample = pd.concat(param_sets)

master_sample.to_csv(outdir+'hmatch_mastersample_'+str(key)+'.csv',index=False)
