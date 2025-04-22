import os
import numpy as np
import xarray as xr
import pandas as pd
import gpflow
import tensorflow as tf
from sklearn.metrics import r2_score
import sys

utils_dir = os.path.dirname(__file__)

def make_preds(nsamp,nrep,usample,biomes,metrics,emulator_path,bpfts):
    samples={p:[np.random.rand(nsamp,15) for i in range(nrep)] for p in range(16)}
    samples['univ']=[np.tile(usample,[nsamp,1]) for i in range(nrep)]

    preds={p:{b:{} for b in biomes[1:]} for p in ['pred','var']}
    for b in biomes[1:]:
        for v in metrics:
            loaded_emulator = tf.saved_model.load(emulator_path + v + '/' + b)
            ys=[]
            yvs=[]
            for i in range(nrep):
                s=np.hstack([samples[p][i] for p in ['univ',*bpfts[b]]])
                y, yv = loaded_emulator.predict(s)
                ys.append(y.numpy().ravel())
                yvs.append(yv.numpy().ravel())
            preds['pred'][b][v]=np.hstack(ys)
            preds['var'][b][v]=np.hstack(yvs)
    return preds

def calc_imp(preds,biomes,obs):
    implaus={b:{} for b in preds['pred']}
    for b in preds['pred']:
        i=biomes.index(b)
        for v in preds['pred']['Tundra']:
            dvs=np.array([v for v in obs.data_vars])
            ix=[v in x.lower() for x in dvs]
            v1,v2=dvs[ix]
            obs_mean=obs[v1][i].values
            obs_var=obs[v2][i].values**2
            
            zimp=np.abs(preds['pred'][b][v] - obs_mean) / np.sqrt(preds['var'][b][v]+ obs_var)
            implaus[b][v]=zimp
    return implaus

def make_da(x):
    da=xr.DataArray([[[x[b][v] for v in x['Tundra']] for b in x]],
                    dims=['usamp','biome','metric','psamp'])
    return da

def get_defaults():
    params_lhc = pd.read_csv('/glade/work/linnia/CLM6-PPE/ctsm6_lhc/ctsm6lhc_11262024.txt').drop(columns='member')
    pft_params   = ['kmax','psi50','jmaxb0','slatop','lmr_intercept_atkin',
                    'medlynslope','medlynintercept','froot_leaf','leafcn','leaf_long',
                    'KCN','dleaf','r_mort','fsr_pft','xl']
    pftix=np.array([p in pft_params for p in params_lhc.columns])
    u_params = params_lhc.columns[~pftix]
    f='/glade/u/home/linnia/ctsm6_ppe/analysis_lhc/wave2_biome/calibrate/default_parameter_settings.csv'
    default_params = pd.read_csv(f)
    default_params = default_params.set_index('name')
    tmp = default_params.transpose()
    default = tmp.loc[['default norm']]
    
    return pft_params,u_params,default

def get_bpfts():
    bpfts={'NaN': [],
     'Tropical rainforest': [4],
     'Tropical savanna': [4, 14],
     'Subtropical savanna': [4, 6, 14],
     'Broadleaf evergreen temperate tree': [5, 13, 14],
     'Grasslands': [13, 14],
     'Shrubland': [10, 13, 14],
     'Mixed deciduous temperate forest': [1, 7, 13, 14],
     'Conifer forest': [1, 2, 13, 14],
     'Siberian larch': [3, 11, 12],
     'Boreal forest': [2, 11, 12],
     'Broadleaf deciduous boreal trees': [2, 8, 12, 13],
     'Boreal shrubland': [12],
     'Tundra': [11, 12]}
    return bpfts








