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
files = np.sort(glob.glob(d+str(key)+'_Atree*'+str(b)+'.csv'))
Atree_biome4 = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

# B tree
d = '/glade/work/linnia/CLM6-PPE/ctsm6_wave1/NROY/'
b=10
files = np.sort(glob.glob(d+str(key)+'_Btree*'+str(b)+'.csv'))
Btree_biome10 = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

#############################################
########################################################
# Find unique universal sets that are in both trees
uA_unique = Atree_biome4[u_params].drop_duplicates()
uB_unique = Btree_biome10[u_params].drop_duplicates()
uset_intersection = uA_unique.merge(uB_unique, how='inner')

biome = 'Conifer forest'
b = 8
nparameters = 41+15*4

pft2_param_names = [f"{param}_{2}" for param in pft_params]
pft13_param_names = [f"{param}_{13}" for param in psample.columns]
pft14_param_names = [f"{param}_{14}" for param in psample.columns]

loaded_emulator = tf.saved_model.load(emulator_dir + biome)

obs_mean = obs.LAI_mean.sel(biome=b).values
obs_var = obs.LAI_stdev.sel(biome=b).values**2

for ux in range(len(uset_intersection)):
    Atree_subset = Atree_biome4.merge(uset_intersection.loc[[ux]], how='inner')
    cols = np.concatenate([u_params,pft13_param_names,pft14_param_names]).tolist()
    Atree_subset = Atree_subset[cols].drop_duplicates()
    Btree_subset = Btree_biome10.merge(uset_intersection.loc[[ux]], how='inner')
    
    for a in np.random.choice(len(Atree_subset),size=min(len(Atree_subset),100),replace=False):
        cols = np.concatenate([u_params,pft13_param_names,pft14_param_names]).tolist()
        start_array = np.tile(Atree_subset[cols].iloc[a].values,(n_psamp,1))
        Astart = pd.DataFrame(start_array, columns=cols)
    
        u = Astart[u_params]
        p13 = Astart[pft13_param_names]
        p14 = Astart[pft14_param_names]

        for b in np.random.choice(len(Btree_subset),size=min(len(Btree_subset),100),replace=False):

            cols = np.concatenate([u_params,pft2_param_names]).tolist()
            start_array = np.tile(Btree_subset[cols].iloc[b].values,(n_psamp,1))
            Bstart = pd.DataFrame(start_array, columns=cols)

            p2 = Bstart[pft2_param_names]

            p1 = psample.values
            sample = np.concatenate([u,p1,p2,p13,p14],axis=1)
        
            # emulate sample
            y_pred, y_pred_var = loaded_emulator.predict(sample)
            
            # calculate implausibility
            I = np.abs(y_pred.numpy().flatten()-obs_mean)/ np.sqrt(obs_var + y_pred_var.numpy().flatten())
            ix = np.where(I<2)[0]
            if (ux ==0):
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

pft1_param_names = [f"{param}_{1}" for param in psample.columns]
pft13_param_names = [f"{param}_{13}" for param in psample.columns]
pft14_param_names = [f"{param}_{14}" for param in psample.columns]
cols = np.concatenate([u_params,pft1_param_names,pft13_param_names,pft14_param_names]).tolist()

obs_mean = obs.LAI_mean.sel(biome=b).values
obs_var = obs.LAI_stdev.sel(biome=b).values**2

loaded_emulator = tf.saved_model.load(emulator_dir + biome)

nx = np.random.choice(len(biome8_sample),size=min(len(biome8_sample),100),replace=False)
for n,i in enumerate(nx):
    start_array = np.tile(biome8_sample[cols].iloc[i].values,(n_psamp,1))
    start = pd.DataFrame(start_array, columns=cols)
    u = start[u_params]
    p1 = start[pft1_param_names]
    p13 = start[pft13_param_names]
    p14 = start[pft14_param_names]

    p7 = psample.values
    sample = np.concatenate([u,p1,p7,p13,p14],axis=1)

    # emulate sample
    y_pred, y_pred_var = loaded_emulator.predict(sample)
    
    # calculate implausibility
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
columns = np.concatenate((u_params, pft1_param_names, pft13_param_names, pft14_param_names))
biome7_unique = biome7_sample[columns].drop_duplicates()

biome8_sample = biome8_sample.merge(biome7_unique, on=columns.tolist(), how='inner')

#############################################
# intersection of biome 8 and Btree
cols = np.concatenate([u_params,pft2_param_names]).tolist()
b8_unique = biome8_sample[cols].drop_duplicates()
uB_unique = Btree_biome10[cols].drop_duplicates()
uset_intersection = b8_unique.merge(uB_unique, how='inner')

biome = 'Broadleaf deciduous boreal trees' # 2,8,12,13
b = 11
nparameters = 41+15*4

obs_mean = obs.LAI_mean.sel(biome=b).values
obs_var = obs.LAI_stdev.sel(biome=b).values**2

loaded_emulator = tf.saved_model.load(emulator_dir + biome)

pft2_param_names = [f"{param}_{2}" for param in pft_params]
pft12_param_names = [f"{param}_{12}" for param in pft_params]
pft13_param_names = [f"{param}_{13}" for param in pft_params]

for ux in range(len(uset_intersection)):
    b8_subset = biome8_sample.merge(uset_intersection.loc[[ux]], how='inner')
    cols = np.concatenate([u_params,pft2_param_names,pft13_param_names]).tolist()
    b8_subset = b8_subset[cols].drop_duplicates()
    
    Btree_subset = Btree_biome10.merge(uset_intersection.loc[[ux]], how='inner')
    
    for c in np.random.choice(len(b8_subset),size=min(len(b8_subset),100),replace=False):
        cols = np.concatenate([u_params,pft2_param_names,pft13_param_names]).tolist()
        start_array = np.tile(b8_subset[cols].iloc[c].values,(n_psamp,1))
        Cstart = pd.DataFrame(start_array, columns=cols)
    
        u = Cstart[u_params]
        p2 = Cstart[pft2_param_names]
        p13 = Cstart[pft13_param_names]

        # loop over all Btree subset options for 12 for Cstart of U+p2
        cols = np.concatenate([u_params,pft2_param_names]).tolist()
        Btree_options = Btree_subset.merge(Cstart[cols],how='inner')
        cols = np.concatenate([u_params,pft2_param_names,pft12_param_names]).tolist()
        Btree_options = Btree_options[cols].drop_duplicates()

        for b in np.random.choice(len(Btree_options),size=min(len(Btree_options),100),replace=False):

            cols = np.concatenate([u_params,pft2_param_names,pft12_param_names]).tolist()
            start_array = np.tile(Btree_options[cols].iloc[b].values,(n_psamp,1))
            Bstart = pd.DataFrame(start_array, columns=cols)

            p12 = Bstart[pft12_param_names]

            # introduce new PFT
            p8 = psample.values
            sample = np.concatenate([u,p2,p8,p12,p13],axis=1)

            # emulate full sample (n_psamp)
            
            y_pred, y_pred_var = loaded_emulator.predict(sample)

            # calc implausibility
            I = np.abs(y_pred.numpy().flatten()-obs_mean)/ np.sqrt(obs_var + y_pred_var.numpy().flatten())
            ix = np.where(I<2)[0]
            # save sample
            if (ux ==0):
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

cols = np.concatenate((u_params, pft2_param_names, pft13_param_names))
biome11_unique = biome11_sample[cols].drop_duplicates()
b8_sample = biome8_sample.merge(biome11_unique, on=cols.tolist(), how='inner')
b8_sample.to_csv(outdir+key+"_Ctree_biome8.csv", index=False)
