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
# load Atree and Btree samples

# A tree
d = '/glade/work/linnia/CLM6-PPE/ctsm6_wave1/NROY/'
b=4
files = np.sort(glob.glob(d+'*tree1*'+str(b)+'.csv'))
Atree_biome4 = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

# B tree
d = '/glade/work/linnia/CLM6-PPE/ctsm6_wave1/NROY/'
b=10
files = np.sort(glob.glob(d+'*Btree*'+str(b)+'.csv'))
Btree_biome10 = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

#############################################
# Find unique universal sets that are in both trees
uA_unique = Atree_biome4[u_params].drop_duplicates()
uB_unique = Btree_biome10[u_params].drop_duplicates()
uset_intersection = uA_unique.merge(uB_unique, how='inner')

biome = 'Conifer forest'
b = 8
nparameters = 41+15*4
cols = u_params

obs_mean = obs.LAI_mean.sel(biome=b).values
obs_var = obs.LAI_stdev.sel(biome=b).values**2

uset_ix = np.random.choice(len(uset_intersection),size=min(len(uset_intersection),30),replace=False)
for n,ux in enumerate(uset_ix):
    u = np.tile(uset_intersection.iloc[ux].values,(n_psamp,1))
    Atree_subset = Atree_biome4.merge(uset_intersection.loc[[ux]], on=cols.tolist(), how='inner')
    Btree_subset = Btree_biome10.merge(uset_intersection.loc[[ux]], on=cols.tolist(), how='inner')
    pft2_param_names = [f"{param}_{2}" for param in pft_params]
    for i in np.random.choice(np.shape(Btree_subset)[0],size=min(np.shape(Btree_subset)[0],10),replace=False):
        p2 = np.tile(Btree_subset[pft2_param_names].iloc[i].values,(n_psamp,1))
        for j in np.random.choice(np.shape(Atree_subset)[0],size=min(np.shape(Atree_subset)[0],10),replace=False):
            pft13_param_names = [f"{param}_{13}" for param in pft_params]
            p13 = np.tile(Atree_subset[pft13_param_names].iloc[j].values,(n_psamp,1))
            pft14_param_names = [f"{param}_{14}" for param in pft_params]
            p14 = np.tile(Atree_subset[pft14_param_names].iloc[j].values,(n_psamp,1))

            p1 = psample.values
            sample = np.concatenate([u,p1,p2,p13,p14],axis=1)
        
            loaded_emulator = tf.saved_model.load(emulator_dir + biome)
            y_pred, y_pred_var = loaded_emulator.predict(sample)
            
            I = np.abs(y_pred.numpy().flatten()-obs_mean)/ np.sqrt(obs_var + y_pred_var.numpy().flatten())
            ix = np.where(I<2)[0]
            if (n ==0):
                biome8_samples = sample[ix,:]
            else:
                biome8_samples = np.concatenate((biome8_samples,sample[ix,:]),axis=0)

# create dataframe
pft1_param_names = [f"{param}_{1}" for param in psample.columns]
columns = np.concatenate((u_params,pft1_param_names,pft2_param_names,pft13_param_names,pft14_param_names))
biome8_sample = pd.DataFrame(biome8_samples,columns=columns)
print('biome 8 done')

#############################################
biome = 'Mixed deciduous temperate forest'
b = 7
nparameters = 41+15*4

obs_mean = obs.LAI_mean.sel(biome=b).values
obs_var = obs.LAI_stdev.sel(biome=b).values**2

nx = np.random.choice(np.shape(biome8_sample)[0],size=min(np.shape(biome8_sample)[0],100),replace=False)

for n,i in enumerate(nx):
    u = np.tile(biome8_sample[u_params].iloc[i].values,(n_psamp,1))
    pft1_param_names = [f"{param}_{1}" for param in psample.columns]
    p1 = np.tile(biome8_sample[pft1_param_names].iloc[i].values,(n_psamp,1))
    pft13_param_names = [f"{param}_{13}" for param in psample.columns]
    p13 = np.tile(biome8_sample[pft13_param_names].iloc[i].values,(n_psamp,1))
    pft14_param_names = [f"{param}_{14}" for param in psample.columns]
    p14 = np.tile(biome8_sample[pft14_param_names].iloc[i].values,(n_psamp,1))

    p7 = psample.values
    sample = np.concatenate([u,p1,p7,p13,p14],axis=1)

    loaded_emulator = tf.saved_model.load(emulator_dir + biome)
    y_pred, y_pred_var = loaded_emulator.predict(sample)
    
    I = np.abs(y_pred.numpy().flatten()-obs_mean)/ np.sqrt(obs_var + y_pred_var.numpy().flatten())
    ix = np.where(I<2)[0]
    if (n ==0):
        biome7_samples = sample[ix,:]
    else:
        biome7_samples = np.concatenate((biome7_samples,sample[ix,:]),axis=0)

# create dataframe
pft7_param_names = [f"{param}_{7}" for param in psample.columns]
columns = np.concatenate((u_params,pft1_param_names,pft7_param_names,pft13_param_names,pft14_param_names))
biome7_sample = pd.DataFrame(biome7_samples,columns=columns)
print('biome 7 done')

#############################################
# biome 7 prunes biome 8 (due to subsampling)
pft13_param_names = [f"{param}_{13}" for param in pft_params]
columns = np.concatenate((u_params, pft13_param_names))
biome7_unique = biome7_sample[columns].drop_duplicates()

biome8_sample = biome8_sample.merge(biome7_unique, on=columns.tolist(), how='inner')

#############################################
# intersection of biome 8 and Btree
b8_unique = biome8_sample[u_params].drop_duplicates()
uB_unique = Btree_biome10[u_params].drop_duplicates()
uset_intersection = b8_unique.merge(uB_unique, how='inner')

biome = 'Broadleaf deciduous boreal trees' # 2,8,12,13
b = 11
nparameters = 41+15*4
cols = u_params

obs_mean = obs.LAI_mean.sel(biome=b).values
obs_var = obs.LAI_stdev.sel(biome=b).values**2

pft2_param_names = [f"{param}_{2}" for param in pft_params]
pft12_param_names = [f"{param}_{12}" for param in pft_params]
pft13_param_names = [f"{param}_{13}" for param in pft_params]

uset_ix = np.random.choice(len(uset_intersection),size=min(len(uset_intersection),30),replace=False)
for n,ux in enumerate(uset_ix): # loop over unique universal samples
    u = np.tile(uset_intersection.iloc[ux].values,(n_psamp,1))
    # find all biome8 sets and Btree sets for this universal sample
    b8_subset = biome8_sample.merge(uset_intersection.loc[[ux]], on=cols.tolist(), how='inner')
    Btree_subset = Btree_biome10.merge(uset_intersection.loc[[ux]], on=cols.tolist(), how='inner')

    # for each biome8 set, select the respective p2 and p13 parameters
    for i in np.random.choice(np.shape(b8_subset)[0],size=min(np.shape(b8_subset)[0],10),replace=False):
        p2 = np.tile(b8_subset[pft2_param_names].iloc[i].values,(n_psamp,1))
        p13 = np.tile(b8_subset[pft13_param_names].iloc[i].values,(n_psamp,1))

        # for each Btree set, select the respective p12 parameters
        for j in np.random.choice(np.shape(Btree_subset)[0],size=min(np.shape(Btree_subset)[0],10),replace=False):
            p12 = np.tile(Btree_subset[pft12_param_names].iloc[j].values,(n_psamp,1))

            # introduce new PFT
            p8 = psample.values
            sample = np.concatenate([u,p2,p8,p12,p13],axis=1)

            # emulate full sample (n_psamp)
            loaded_emulator = tf.saved_model.load(emulator_dir + biome)
            y_pred, y_pred_var = loaded_emulator.predict(sample)

            # calc implausibility
            I = np.abs(y_pred.numpy().flatten()-obs_mean)/ np.sqrt(obs_var + y_pred_var.numpy().flatten())
            ix = np.where(I<3)[0]
            # save sample
            if (n ==0):
                biome11_samples = sample[ix,:]
            else:
                biome11_samples = np.concatenate((biome11_samples,sample[ix,:]),axis=0)

# create NROY dataframe
pft8_param_names = [f"{param}_{8}" for param in psample.columns]
columns = np.concatenate((u_params,pft2_param_names,pft8_param_names,pft12_param_names,pft13_param_names))
biome11_sample = pd.DataFrame(biome11_samples,columns=columns)
print('biome 11 done')

#############################################
# Prune up and save
b11_sample = biome11_sample
b11_sample.to_csv(outdir+key+"_Ctree_biome11.csv", index=False)

cols = np.concatenate((u_params, pft13_param_names))
biome11_unique = biome11_sample[cols].drop_duplicates()

b7_sample = biome7_sample.merge(biome11_unique, on=cols.tolist(), how='inner')
b7_sample.to_csv(outdir+key+"_Ctree_biome7.csv", index=False)

b8_sample = biome8_sample.merge(biome11_unique, on=cols.tolist(), how='inner')
b8_sample.to_csv(outdir+key+"_Ctree_biome8.csv", index=False)

