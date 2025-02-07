import numpy as np
import pandas as pd
import xarray as xr
import tensorflow as tf
import sys
import pickle
from pyfunctions import *

# input to script
idx = sys.argv[1]

#############################################
# user mods
#############################################
n_usamp = 100
n_psamp = 1000

pft_ids = [1,2,3,4,5,6,7,8,10,11,12,13,14]
npfts = len(pft_ids)
I_thresh=3
min_npsamp = 10

emulator_dir = '/glade/u/home/linnia/ctsm6_ppe/analysis_lhc/wave1/emulators_pftlai_amax_lhc/'
out_dir = '/glade/work/linnia/CLM6-PPE/ctsm6_lhc/NROY/'


#############################################
# Setup
#############################################
# load obs data
obs_data = xr.open_dataset("../CLMSP_amaxLAI_2003-2015_pftmean.nc")

# load PPE
key = '/glade/work/linnia/CLM6-PPE/ctsm6_lhc/ctsm6lhc_11262024.txt'
params = pd.read_csv(key).drop(columns='member')
n_params = np.shape(params)[1]

dir='/glade/work/linnia/CLM6-PPE/ctsm6_lhc/postp/'
ds = xr.open_dataset(dir+'ctsm6lhc_pftmean_1985-2023.nc')

# drop crashers
ds_clean = ds.where(ds['crashed'] != 1, drop=True)
ix = np.where(ds.crashed==1)[0]
params_clean = params.drop(ix-1)


#############################################
# History Matching
#############################################
# create random sample
sample = create_random_sample(n_usamp,n_psamp,params_clean.columns)

#############################################
# Calculate implausibility

set_I = np.empty((n_usamp*n_psamp,len(pft_ids)))*np.NaN

for p,pft in enumerate(pft_ids):

        loaded_emulator = tf.saved_model.load(emulator_dir + 'pft'+str(pft))
        y_pred, y_pred_var = loaded_emulator.predict(sample)

        m_mn = y_pred.numpy().flatten()
        m_var = y_pred_var.numpy().flatten()
        o_mn = obs_data['Mean'].sel(pft=pft).values
        o_var = obs_data['StDev'].sel(pft=pft).values**2
        
        # calculate implausibility
        set_I[:,p] = np.abs(o_mn-m_mn) / np.sqrt(o_var + m_var)


#############################################
#identify universal sets that have at least n_pftsets NROY sets for all PFTs 
uset_idx = usample_NROY(set_I,n_usamp,n_psamp,min_npsamp,npfts)

#############################################
# select a coresponding PFT set for each PFT and Uset
I_bool = np.where(set_I < 3, True, False)

NROY_sample = np.empty((len(uset_idx),56,npfts))
s = sample.reshape(n_usamp,n_psamp,56)[uset_idx,:,:]
for p,pft in enumerate(pft_ids):
    indices=[]
    m = I_bool[:,p].reshape(n_usamp,n_psamp)[uset_idx,:]
    for row in m:
        nroy_ix = np.where(row)[0]
        pftset_ix = np.random.choice(nroy_ix)
        indices.append(pftset_ix)
        
    NROY_sample[:, :, p] = s[np.arange(len(uset_idx)), indices, :]


#############################################
# create datasets and write to out_dir
ds_nroy = xr.Dataset(
    {"nroy_paramsets": (["ens", "param", "pft"], NROY_sample)},
    coords={
        "ens": range(len(uset_idx)),
        "param": params.columns,
        "pft": pft_ids,},
)

filename = out_dir + "pft_laimax_NROYsets_sample"+str(idx)+".nc"
ds_nroy.to_netcdf(filename)

