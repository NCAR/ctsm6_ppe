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
# work tree 1

biome = 'Tropical rainforest'
b = 1
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

pft_param_names = [f"{param}_{4}" for param in psample.columns]
columns = np.concatenate((usample.columns,pft_param_names))
biome1_sample = pd.DataFrame(sample[ix],columns=columns)

#############################################
biome = 'Tropical savanna'
b = 2
nparameters = 41+15*2

pft4_param_names = [f"{param}_{4}" for param in psample.columns]
cols = np.concatenate([u_params,pft4_param_names]).tolist()

obs_mean = obs.LAI_mean.sel(biome=b).values
obs_var = obs.LAI_stdev.sel(biome=b).values**2

nx = np.random.choice(len(biome1_sample),size=min(len(biome1_sample),1000),replace=False)

for n,i in enumerate(nx):
    start_array = np.tile(biome1_sample[cols].iloc[i].values,(n_psamp,1))
    start = pd.DataFrame(start_array, columns=cols)
    u = start[u_params]
    p4 = start[pft4_param_names]
    p14 = psample.values
    sample = np.concatenate([u,p4,p14],axis=1)
    
    loaded_emulator = tf.saved_model.load(emulator_dir + biome)
    y_pred, y_pred_var = loaded_emulator.predict(sample)
    
    I = np.abs(y_pred.numpy().flatten()-obs_mean)/ np.sqrt(obs_var + y_pred_var.numpy().flatten())
    ix = np.where(I<2)[0]
    if (n ==0):
        biome2_samples = sample[ix,:]
    else:
        biome2_samples = np.concatenate((biome2_samples,sample[ix,:]),axis=0)

# create dataframe
pft14_param_names = [f"{param}_{14}" for param in psample.columns]
columns = np.concatenate((usample.columns,pft4_param_names,pft14_param_names))
biome2_sample = pd.DataFrame(biome2_samples,columns=columns)

#############################################
biome = 'Subtropical savanna'
b = 3
nparameters = 41+15*3

pft4_param_names = [f"{param}_{4}" for param in psample.columns]
pft14_param_names = [f"{param}_{14}" for param in psample.columns]
cols = np.concatenate([u_params,pft4_param_names,pft14_param_names]).tolist()

obs_mean = obs.LAI_mean.sel(biome=b).values
obs_var = obs.LAI_stdev.sel(biome=b).values**2

loaded_emulator = tf.saved_model.load(emulator_dir + biome)

nx = np.random.choice(len(biome2_sample),size=min(len(biome2_sample),1000),replace=False)
for n,i in enumerate(nx):
    start_array = np.tile(biome2_sample[cols].iloc[i].values,(n_psamp,1))
    start = pd.DataFrame(start_array, columns=cols)
    u = start[u_params]
    p4 = start[pft4_param_names]
    p14 = start[pft14_param_names]
    p6 = psample.values
    sample = np.concatenate([u,p4,p6,p14],axis=1)

    # emulate sample
    y_pred, y_pred_var = loaded_emulator.predict(sample)
    
    I = np.abs(y_pred.numpy().flatten()-obs_mean)/ np.sqrt(obs_var + y_pred_var.numpy().flatten())
    ix = np.where(I<2)[0]
    if (n ==0):
        biome3_samples = sample[ix,:]
    else:
        biome3_samples = np.concatenate((biome3_samples,sample[ix,:]),axis=0)

# create dataframe
pft6_param_names = [f"{param}_{6}" for param in psample.columns]
columns = np.concatenate((u_params,pft4_param_names,pft6_param_names,pft14_param_names))
biome3_sample = pd.DataFrame(biome3_samples,columns=columns)

#############################################
biome = 'Grasslands'
b = 5
nparameters = 41+15*2

pft14_param_names = [f"{param}_{14}" for param in psample.columns]
cols = np.concatenate([u_params,pft14_param_names]).tolist()

obs_mean = obs.LAI_mean.sel(biome=b).values
obs_var = obs.LAI_stdev.sel(biome=b).values**2

loaded_emulator = tf.saved_model.load(emulator_dir + biome)

nx = np.random.choice(len(biome3_sample),size=min(len(biome3_sample),1000),replace=False)
for n,i in enumerate(nx):
    start_array = np.tile(biome3_sample[cols].iloc[i].values,(n_psamp,1))
    start = pd.DataFrame(start_array, columns=cols)
    u = start[u_params]
    p14 = start[pft14_param_names]
    
    p13 = psample.values
    sample = np.concatenate([u,p13,p14],axis=1)

    # emulate sample
    y_pred, y_pred_var = loaded_emulator.predict(sample)
    
    I = np.abs(y_pred.numpy().flatten()-obs_mean)/ np.sqrt(obs_var + y_pred_var.numpy().flatten())
    ix = np.where(I<2)[0]
    if (n ==0):
        biome5_samples = sample[ix,:]
    else:
        biome5_samples = np.concatenate((biome5_samples,sample[ix,:]),axis=0)

# create dataframe
pft13_param_names = [f"{param}_{13}" for param in psample.columns]
columns = np.concatenate((u_params,pft13_param_names,pft14_param_names))
biome5_sample = pd.DataFrame(biome5_samples,columns=columns)

#############################################
biome = 'Shrubland'
b = 6
nparameters = 41+15*3

pft13_param_names = [f"{param}_{13}" for param in psample.columns]
pft14_param_names = [f"{param}_{14}" for param in psample.columns]
cols = np.concatenate([u_params,pft13_param_names,pft14_param_names]).tolist()

obs_mean = obs.LAI_mean.sel(biome=b).values
obs_var = obs.LAI_stdev.sel(biome=b).values**2

loaded_emulator = tf.saved_model.load(emulator_dir + biome)

nx = np.random.choice(len(biome5_sample),size=min(len(biome5_sample),1000),replace=False)
for n,i in enumerate(nx):
    start_array = np.tile(biome5_sample[cols].iloc[i].values,(n_psamp,1))
    start = pd.DataFrame(start_array, columns=cols)
    u = start[u_params]
    p13 = start[pft13_param_names]
    p14 = start[pft14_param_names]
    
    p10 = psample.values
    sample = np.concatenate([u,p10,p13,p14],axis=1)

    # emulate sample
    y_pred, y_pred_var = loaded_emulator.predict(sample)
    
    I = np.abs(y_pred.numpy().flatten()-obs_mean)/ np.sqrt(obs_var + y_pred_var.numpy().flatten())
    ix = np.where(I<2)[0]
    if (n ==0):
        biome6_samples = sample[ix,:]
    else:
        biome6_samples = np.concatenate((biome6_samples,sample[ix,:]),axis=0)

# create dataframe
pft10_param_names = [f"{param}_{10}" for param in psample.columns]
pft13_param_names = [f"{param}_{13}" for param in psample.columns]
pft14_param_names = [f"{param}_{14}" for param in psample.columns]
columns = np.concatenate((u_params,pft10_param_names,pft13_param_names,pft14_param_names))
biome6_sample = pd.DataFrame(biome6_samples,columns=columns)

#############################################
biome = 'Broadleaf evergreen temperate tree'
b = 4
nparameters = 41+15*3

pft13_param_names = [f"{param}_{13}" for param in psample.columns]
pft14_param_names = [f"{param}_{14}" for param in psample.columns]
cols = np.concatenate([u_params,pft13_param_names,pft14_param_names]).tolist()

obs_mean = obs.LAI_mean.sel(biome=b).values
obs_var = obs.LAI_stdev.sel(biome=b).values**2

loaded_emulator = tf.saved_model.load(emulator_dir + biome)

nx = np.random.choice(len(biome6_sample),size=min(len(biome6_sample),1000),replace=False)
for n,i in enumerate(nx):
    start_array = np.tile(biome6_sample[cols].iloc[i].values,(n_psamp,1))
    start = pd.DataFrame(start_array, columns=cols)
    u = start[u_params]
    p13 = start[pft13_param_names]
    p14 = start[pft14_param_names]
    
    p5 = psample.values
    sample = np.concatenate([u,p5,p13,p14],axis=1)

    # emulate sample
    y_pred, y_pred_var = loaded_emulator.predict(sample)

    # calc implausibility
    I = np.abs(y_pred.numpy().flatten()-obs_mean)/ np.sqrt(obs_var + y_pred_var.numpy().flatten())
    ix = np.where(I<2)[0]
    if (n ==0):
        biome4_samples = sample[ix,:]
    else:
        biome4_samples = np.concatenate((biome4_samples,sample[ix,:]),axis=0)

# create dataframe
pft5_param_names = [f"{param}_{5}" for param in psample.columns]
columns = np.concatenate((u_params,pft5_param_names,pft13_param_names,pft14_param_names))
biome4_sample = pd.DataFrame(biome4_samples,columns=columns)

#############################################
# prune up and save

# biome 4 PFT 5
b4_sample = biome4_sample
b4_sample.to_csv(outdir+key+"_Atree_b4.csv", index=False)

cols = np.concatenate((u_params, pft13_param_names, pft14_param_names))
biome4_unique = biome4_sample[cols].drop_duplicates()

# biome 6 PFT 10
b6_sample = biome6_sample.merge(biome4_unique, on=cols.tolist(), how='inner')
b6_sample.to_csv(outdir+key+"_Atree_b6.csv", index=False)

# biome 5 PFT 13
b5_sample = biome5_sample.merge(biome4_unique, on=cols.tolist(), how='inner')
b5_sample.to_csv(outdir+key+"_Atree_b5.csv", index=False)

# biome 3 PFT 6
cols = np.concatenate((u_params, pft14_param_names))
biome4_unique = biome4_sample[cols].drop_duplicates()
b3_sample = biome3_sample.merge(biome4_unique, on=cols.tolist(), how='inner')
b3_sample.to_csv(outdir+key+"_Atree_b3.csv", index=False)

# biome 2 PFT 14
cols = np.concatenate((u_params, pft4_param_names, pft14_param_names))
unique = b3_sample[cols].drop_duplicates()
b2_sample = biome2_sample.merge(unique, on=cols.tolist(), how='inner')
b2_sample.to_csv(outdir+key+"_Atree_b2.csv", index=False)

# biome 1 PFT 4
cols = np.concatenate((u_params, pft4_param_names))
biome2_unique = b2_sample[cols].drop_duplicates()
b1_sample = biome1_sample.merge(biome2_unique, on=cols.tolist(), how='inner')
b1_sample.to_csv(outdir+key+"_Atree_b1.csv", index=False)

