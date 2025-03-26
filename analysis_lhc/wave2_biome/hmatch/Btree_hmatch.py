import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import os
import pickle
import gpflow

import sys
sys.path.append('/glade/u/home/linnia/ctsm6_ppe/')
from utils.pyfunctions import *
utils_path = '/glade/u/home/linnia/ctsm6_ppe/utils/'

# input to script
key = str(sys.argv[1])

#############################################
# user mods
#############################################

emulator_dir = '/glade/u/home/linnia/ctsm6_ppe/analysis_lhc/wave2_biome/emulators_biomelai/'
outdir = '/glade/work/linnia/CLM6-PPE/ctsm6_wave1/NROY/'

#############################################
# Setup
#############################################
# load obs data
obs = xr.open_dataset('../wave2_obsStatistics_sudokuBiomes.nc')

# setup
params_lhc = pd.read_csv('/glade/work/linnia/CLM6-PPE/ctsm6_lhc/ctsm6lhc_11262024.txt').drop(columns='member')

pft_params   = ['kmax','psi50','jmaxb0','slatop','lmr_intercept_atkin',
                'medlynslope','medlynintercept','froot_leaf','leafcn','leaf_long',
                'KCN','dleaf','r_mort','fsr_pft','xl']
pftix=np.array([p in pft_params for p in params_lhc.columns])
u_params = params_lhc.columns[~pftix]

file='/glade/u/home/linnia/ctsm6_ppe/utils/sudoku_biomes.nc'
biomes=xr.open_dataset(file)
biome_names = biomes.biome_names.values

import pickle
with open("/glade/u/home/linnia/ctsm6_ppe/utils/biome_pft_dict.pkl", "rb") as f:
    biome_pfts = pickle.load(f)


#############################################
# History Matching
#############################################
# load subsample
n_usamp = 10
n_psamp = 100
usample_all = pd.read_csv("universal_samples.csv")
psample_all = pd.read_csv("pft_samples.csv")


usample = usample_all.iloc[int(key)*n_usamp:int(key)*n_usamp+n_usamp]
psample = psample_all.iloc[int(key)*n_psamp:int(key)*n_psamp+n_psamp]

#############################################
# work tree 2

biome = 'Boreal shrubland'
b = 12
nparameters = 41+15*1

obs_mean = obs.LAI_mean.sel(biome=b).values
obs_var = obs.LAI_stdev.sel(biome=b).values**2

u = np.repeat(usample.values,repeats=n_psamp,axis=0)
p = np.tile(psample.values, (n_usamp, 1))
sample = np.concatenate([u,p],axis=1)

loaded_emulator = tf.saved_model.load(emulator_dir + biome)
y_pred, y_pred_var = loaded_emulator.predict(sample)

I = np.abs(y_pred.numpy().flatten()-obs_mean)/ np.sqrt(obs_var + y_pred_var.numpy().flatten())
ix = np.where(I<2)[0]

pft_param_names = [f"{param}_{12}" for param in psample.columns]
columns = np.concatenate((usample.columns,pft_param_names))
biome12_sample = pd.DataFrame(sample[ix],columns=columns)

#############################################
biome = 'Tundra'
b = 13
nparameters = 41+15*2

pft12_param_names = [f"{param}_{12}" for param in psample.columns]
cols = np.concatenate([u_params,pft12_param_names]).tolist()

obs_mean = obs.LAI_mean.sel(biome=b).values
obs_var = obs.LAI_stdev.sel(biome=b).values**2

nx = np.random.choice(len(biome12_sample),size=min(len(biome12_sample),100),replace=False)
    
for n,i in enumerate(nx):
    start_array = np.tile(biome12_sample[cols].iloc[i].values,(n_psamp,1))
    start = pd.DataFrame(start_array, columns=cols)
    u = start[u_params]
    p12 = start[pft12_param_names]
    p11 = psample.values
    sample = np.concatenate([u,p11,p12],axis=1)
    
    loaded_emulator = tf.saved_model.load(emulator_dir + biome)
    y_pred, y_pred_var = loaded_emulator.predict(sample)
    
    I = np.abs(y_pred.numpy().flatten()-obs_mean)/ np.sqrt(obs_var + y_pred_var.numpy().flatten())
    ix = np.where(I<2)[0]
    if (n ==0):
        biome13_samples = sample[ix,:]
    else:
        biome13_samples = np.concatenate((biome13_samples,sample[ix,:]),axis=0)

# create dataframe
pft11_param_names = [f"{param}_{11}" for param in psample.columns]
columns = np.concatenate((usample.columns,pft11_param_names,pft12_param_names))
biome13_sample = pd.DataFrame(biome13_samples,columns=columns)

#############################################
biome = 'Siberian larch'
b = 9
nparameters = 41+15*3

pft11_param_names = [f"{param}_{11}" for param in psample.columns]
pft12_param_names = [f"{param}_{12}" for param in psample.columns]
cols = np.concatenate([u_params,pft11_param_names,pft12_param_names]).tolist()

obs_mean = obs.LAI_mean.sel(biome=b).values
obs_var = obs.LAI_stdev.sel(biome=b).values**2

nx = np.random.choice(len(biome13_sample),size=min(len(biome13_sample),100),replace=False)
    
for n,i in enumerate(nx):
    start_array = np.tile(biome13_sample[cols].iloc[i].values,(n_psamp,1))
    start = pd.DataFrame(start_array, columns=cols)

    u = start[u_params]
    p11 = start[pft11_param_names]
    p12 = start[pft12_param_names]
    p3 = psample.values
    sample = np.concatenate([u,p3,p11,p12],axis=1)

    loaded_emulator = tf.saved_model.load(emulator_dir + biome)
    y_pred, y_pred_var = loaded_emulator.predict(sample)
    
    I = np.abs(y_pred.numpy().flatten()-obs_mean)/ np.sqrt(obs_var + y_pred_var.numpy().flatten())
    ix = np.where(I<2)[0]
    if (n ==0):
        biome9_samples = sample[ix,:]
    else:
        biome9_samples = np.concatenate((biome9_samples,sample[ix,:]),axis=0)

# create dataframe
pft3_param_names = [f"{param}_{3}" for param in psample.columns]
columns = np.concatenate((u_params,pft3_param_names,pft11_param_names,pft12_param_names))
biome9_sample = pd.DataFrame(biome9_samples,columns=columns)

#############################################
biome = 'Boreal forest'
b = 10
nparameters = 41+15*3

pft11_param_names = [f"{param}_{11}" for param in psample.columns]
pft12_param_names = [f"{param}_{12}" for param in psample.columns]
cols = np.concatenate([u_params,pft11_param_names,pft12_param_names]).tolist()

obs_mean = obs.LAI_mean.sel(biome=b).values
obs_var = obs.LAI_stdev.sel(biome=b).values**2

nx = np.random.choice(len(biome9_sample),size=min(len(biome9_sample),100),replace=False)
    
for n,i in enumerate(nx):
    start_array = np.tile(biome9_sample[cols].iloc[i].values,(n_psamp,1))
    start = pd.DataFrame(start_array, columns=cols)

    u = start[u_params]
    p11 = start[pft11_param_names]
    p12 = start[pft12_param_names]
    
    p2 = psample.values
    sample = np.concatenate([u,p2,p11,p12],axis=1)

    loaded_emulator = tf.saved_model.load(emulator_dir + biome)
    y_pred, y_pred_var = loaded_emulator.predict(sample)
    
    I = np.abs(y_pred.numpy().flatten()-obs_mean)/ np.sqrt(obs_var + y_pred_var.numpy().flatten())
    ix = np.where(I<2)[0]
    if (n ==0):
        biome10_samples = sample[ix,:]
    else:
        biome10_samples = np.concatenate((biome10_samples,sample[ix,:]),axis=0)

# create dataframe
pft2_param_names = [f"{param}_{2}" for param in psample.columns]
columns = np.concatenate((u_params,pft2_param_names,pft11_param_names,pft12_param_names))
biome10_sample = pd.DataFrame(biome10_samples,columns=columns)


#############################################
# Prune up and save

b10_sample = biome10_sample
b10_sample.to_csv(outdir+key+"_Btree_biome10.csv", index=False)

cols = np.concatenate((u_params, pft11_param_names, pft12_param_names))
unique = biome10_sample[cols].drop_duplicates()

b9_sample = biome9_sample.merge(unique, on=cols.tolist(), how='inner')
b9_sample.to_csv(outdir+key+"_Btree_biome9.csv", index=False)

b13_sample = biome13_sample.merge(unique, on=cols.tolist(), how='inner')
b13_sample.to_csv(outdir+key+"_Btree_biome13.csv", index=False)

cols = np.concatenate((u_params, pft12_param_names))
unique = biome10_sample[cols].drop_duplicates()

b12_sample = biome12_sample.merge(unique, on=cols.tolist(), how='inner')
b12_sample.to_csv(outdir+key+"_Btree_biome12.csv", index=False)
