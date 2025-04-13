import os
import numpy as np
import xarray as xr
import cftime
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import glob
import dask

import gpflow
import tensorflow as tf
from sklearn.metrics import r2_score


utils_dir = os.path.dirname(__file__)

# =======================================================

def create_master_sample(b_samples,pft_param_names):
    master_sample = b_samples[3].copy()
    master_sample[pft_param_names[13]] = b_samples[5][pft_param_names[13]].values
    master_sample[pft_param_names[10]] = b_samples[6][pft_param_names[10]].values
    master_sample[pft_param_names[5]] = b_samples[4][pft_param_names[5]].values
    master_sample[pft_param_names[2]] = b_samples[10][pft_param_names[2]].values
    master_sample[pft_param_names[11]] = b_samples[10][pft_param_names[11]].values
    master_sample[pft_param_names[12]] = b_samples[10][pft_param_names[12]].values
    master_sample[pft_param_names[3]] = b_samples[9][pft_param_names[3]].values
    master_sample[pft_param_names[1]] = b_samples[7][pft_param_names[1]].values
    master_sample[pft_param_names[7]] = b_samples[7][pft_param_names[7]].values
    master_sample[pft_param_names[8]] = b_samples[11][pft_param_names[8]].values
    return master_sample

#############################################

def create_biome_sample(b,previous_sample,pft_param_names,u_params,n_psamp,psample):
    if (b == 2):
        u = np.tile(previous_sample[u_params].values,(n_psamp,1))
        p4 = np.tile(previous_sample[pft_param_names[4]].values,(n_psamp,1))
        p14 = psample.values
        columns = np.concatenate((u_params,pft_param_names[4],pft_param_names[14]))
        sample = pd.DataFrame(np.concatenate([u,p4,p14],axis=1),columns=columns)

    if (b == 3):
        u = np.tile(previous_sample[u_params].values,(n_psamp,1))
        p4 = np.tile(previous_sample[pft_param_names[4]].values,(n_psamp,1))
        p14 = np.tile(previous_sample[pft_param_names[14]].values,(n_psamp,1))
        p6 = psample.values
        columns = np.concatenate((u_params,pft_param_names[4],pft_param_names[6],pft_param_names[14]))
        sample = pd.DataFrame(np.concatenate([u,p4,p6,p14],axis=1),columns=columns)

    if (b == 5):
        u = np.tile(previous_sample[u_params].values,(n_psamp,1))
        p14 = np.tile(previous_sample[pft_param_names[14]].values,(n_psamp,1))
        p13 = psample.values
        columns = np.concatenate((u_params,pft_param_names[13],pft_param_names[14]))
        sample = pd.DataFrame(np.concatenate([u,p13,p14],axis=1),columns=columns)

    if (b == 6) | (b == 4):
        u = np.tile(previous_sample[u_params].values,(n_psamp,1))
        p13 = np.tile(previous_sample[pft_param_names[13]].values,(n_psamp,1))
        p14 = np.tile(previous_sample[pft_param_names[14]].values,(n_psamp,1))
        pnew = psample.values
        if (b==6):
            columns = np.concatenate((u_params,pft_param_names[10],pft_param_names[13],pft_param_names[14]))
        elif (b==4):
            columns = np.concatenate((u_params,pft_param_names[5],pft_param_names[13],pft_param_names[14]))
        sample = pd.DataFrame(np.concatenate([u,pnew,p13,p14],axis=1),columns=columns)

    if (b == 12):
        u = np.tile(previous_sample[u_params].values,(n_psamp,1))
        pnew = psample.values
        columns = np.concatenate((u_params,pft_param_names[12]))
        sample = pd.DataFrame(np.concatenate([u,pnew],axis=1),columns=columns)

    if (b == 13):
        u = np.tile(previous_sample[u_params].values,(n_psamp,1))
        p12 = np.tile(previous_sample[pft_param_names[12]].values,(n_psamp,1))
        p11 = psample.values
        columns = np.concatenate((u_params,pft_param_names[11],pft_param_names[12]))
        sample = pd.DataFrame(np.concatenate([u,p11,p12],axis=1),columns=columns)

    if (b == 9) | (b == 10):
        u = np.tile(previous_sample[u_params].values,(n_psamp,1))
        p11 = np.tile(previous_sample[pft_param_names[11]].values,(n_psamp,1))
        p12 = np.tile(previous_sample[pft_param_names[12]].values,(n_psamp,1))
        pnew = psample.values
        if (b == 9):
            columns = np.concatenate((u_params,pft_param_names[3],pft_param_names[11],pft_param_names[12]))
        elif (b == 10):
            columns = np.concatenate((u_params,pft_param_names[2],pft_param_names[11],pft_param_names[12]))
        sample = pd.DataFrame(np.concatenate([u,pnew,p11,p12],axis=1),columns=columns)

    if (b == 8):
        u = np.tile(previous_sample[u_params].values,(n_psamp,1))
        p13 = np.tile(previous_sample[pft_param_names[13]].values,(n_psamp,1))
        p14 = np.tile(previous_sample[pft_param_names[14]].values,(n_psamp,1))
        p2 = np.tile(previous_sample[pft_param_names[2]].values,(n_psamp,1))
        p1 = psample.values
        columns = np.concatenate((u_params,pft_param_names[1],pft_param_names[2],pft_param_names[13],pft_param_names[14]))
        sample = pd.DataFrame(np.concatenate([u,p1,p2,p13,p14],axis=1),columns=columns)

    if (b == 7):
        u = np.tile(previous_sample[u_params].values,(n_psamp,1))
        p1 = np.tile(previous_sample[pft_param_names[1]].values,(n_psamp,1))
        p13 = np.tile(previous_sample[pft_param_names[13]].values,(n_psamp,1))
        p14 = np.tile(previous_sample[pft_param_names[14]].values,(n_psamp,1))
        p7 = psample.values
        columns = np.concatenate((u_params,pft_param_names[1],pft_param_names[7],pft_param_names[13],pft_param_names[14]))
        sample = pd.DataFrame(np.concatenate([u,p1,p7,p13,p14],axis=1),columns=columns)

    if (b == 11):
        u = np.tile(previous_sample[u_params].values,(n_psamp,1))
        p13 = np.tile(previous_sample[pft_param_names[13]].values,(n_psamp,1))
        p12 = np.tile(previous_sample[pft_param_names[12]].values,(n_psamp,1))
        p2 = np.tile(previous_sample[pft_param_names[2]].values,(n_psamp,1))
        p8 = psample.values
        columns = np.concatenate((u_params,pft_param_names[2],pft_param_names[8],pft_param_names[12],pft_param_names[13]))
        sample = pd.DataFrame(np.concatenate([u,p2,p8,p12,p13],axis=1),columns=columns)

    return sample

#############################################

def hmatch_biome(b, biome_name, sample, emulator_path, obs, minimize=False):
    
    # LAI
    loaded_emulator = tf.saved_model.load(emulator_path+'lai/' + biome_name)
    y_pred, y_pred_var = loaded_emulator.predict(sample.values)
    obs_mean, obs_var = obs.LAI_mean.sel(biome=b).values, obs.LAI_stdev.sel(biome=b).values**2
    lai_I = np.abs(y_pred.numpy().flatten() - obs_mean) / np.sqrt(y_pred_var.numpy().flatten() + obs_var)
    
    # GPP
    loaded_emulator = tf.saved_model.load(emulator_path+'gpp/' + biome_name)
    y_pred, y_pred_var = loaded_emulator.predict(sample.values)
    obs_mean, obs_var = obs.GPP_mean.sel(biome=b).values, obs.GPP_stdev.sel(biome=b).values**2
    gpp_I = np.abs(y_pred.numpy().flatten() - obs_mean) / np.sqrt(y_pred_var.numpy().flatten() + obs_var)
    
    # biomass
    loaded_emulator = tf.saved_model.load(emulator_path+'biomass/' + biome_name)
    y_pred, y_pred_var = loaded_emulator.predict(sample.values)
    obs_mean, obs_var = obs.biomassC_mean.sel(biome=b).values, obs.biomassC_stdev.sel(biome=b).values**2
    biomass_I = np.abs(y_pred.numpy().flatten() - obs_mean) / np.sqrt(y_pred_var.numpy().flatten() + obs_var)

    mask = (lai_I < 3) & (gpp_I < 3) & (biomass_I < 3)
    plausible_ix = np.where(mask)[0]
    if len(plausible_ix) == 0: # if there are no plausible_ix samples then continue
        selected_sample = pd.DataFrame({np.nan})
    elif minimize == 1:
        # Find the index among plausible_ix with the minimum total implausibility score.
        sum_I = lai_I + gpp_I + biomass_I
        min_index = np.argmin(sum_I[plausible_ix])
        ix = plausible_ix[min_index]
        selected_sample = sample.iloc[[ix]]
    else: # take one random sample from plausible_ix
        ix = np.random.choice(plausible_ix) 
        selected_sample = sample.iloc[[ix]]
    
    return selected_sample


#############################################

def calibration_tree(usample,psample,n_psamp,u_params,pft_param_names,emulator_path,obs,biome_configs,minimize=False,):

    ######
    # Atree
    b = 1 # 'Tropical rainforest'
    biome_name = biome_configs[b]['name']
    u = np.repeat(usample.values,repeats=n_psamp,axis=0)
    s = np.concatenate([u,psample.values],axis=1)
    columns = np.concatenate((u_params,pft_param_names[4]))
    sample = pd.DataFrame(s,columns=columns)
    biome1_sample = hmatch_biome(b, biome_name, sample, emulator_path, obs, minimize)
    if biome1_sample.isna().all().all():
        return pd.DataFrame({np.nan})
    
    ####
    b = 2   # 'Tropical savanna'
    biome_name = biome_configs[b]['name']
    sample = create_biome_sample(b,biome1_sample,pft_param_names,u_params,n_psamp,psample)
    biome2_sample = hmatch_biome(b, biome_name,sample, emulator_path, obs, minimize)
    if biome2_sample.isna().all().all():
        return pd.DataFrame({np.nan})
        
    ####
    b = 3   # 'Subtropical savanna'
    biome_name = biome_configs[b]['name']
    sample = create_biome_sample(b,biome2_sample,pft_param_names,u_params,n_psamp,psample)
    biome3_sample = hmatch_biome(b, biome_name, sample, emulator_path, obs, minimize)
    if biome3_sample.isna().all().all():
        return pd.DataFrame({np.nan})

    ####
    b = 5  # 'Grasslands'
    biome_name = biome_configs[b]['name']
    sample = create_biome_sample(b,biome3_sample,pft_param_names,u_params,n_psamp,psample)
    biome5_sample = hmatch_biome(b, biome_name, sample, emulator_path, obs, minimize)
    if biome5_sample.isna().all().all():
        return pd.DataFrame({np.nan})
    
    ####
    b = 6  # 'Shrubland'
    biome_name = biome_configs[b]['name']
    sample = create_biome_sample(b,biome5_sample,pft_param_names,u_params,n_psamp,psample)
    biome6_sample = hmatch_biome(b, biome_name, sample, emulator_path, obs, minimize)
    if biome6_sample.isna().all().all():
        return pd.DataFrame({np.nan})
    
    ####
    b = 4   # 'Broadleaf evergreen temperate tree'
    biome_name = biome_configs[b]['name']
    sample = create_biome_sample(b,biome6_sample,pft_param_names,u_params,n_psamp,psample)
    biome4_sample = hmatch_biome(b, biome_name, sample, emulator_path, obs, minimize)
    if biome4_sample.isna().all().all():
        return pd.DataFrame({np.nan})

    ###########
    # Btree
    b = 12 # 'Boreal shrubland'
    biome_name = biome_configs[b]['name']
    sample = create_biome_sample(b,biome1_sample,pft_param_names,u_params,n_psamp,psample)
    biome12_sample = hmatch_biome(b, biome_name, sample, emulator_path, obs, minimize)
    if biome12_sample.isna().all().all():
        return pd.DataFrame({np.nan})
    
    ####
    b = 13 # 'Tundra'
    biome_name = biome_configs[b]['name']
    sample = create_biome_sample(b,biome12_sample,pft_param_names,u_params,n_psamp,psample)
    biome13_sample = hmatch_biome(b, biome_name, sample, emulator_path, obs, minimize)
    if biome13_sample.isna().all().all():
        return pd.DataFrame({np.nan})
    
    ####
    b = 9  # 'Siberian larch'
    biome_name = biome_configs[b]['name']
    sample = create_biome_sample(b,biome13_sample,pft_param_names,u_params,n_psamp,psample)
    biome9_sample = hmatch_biome(b, biome_name, sample, emulator_path, obs, minimize)
    if biome9_sample.isna().all().all():
        return pd.DataFrame({np.nan})
        
    ####
    b = 10 # 'Boreal forest'
    biome_name = biome_configs[b]['name']
    sample = create_biome_sample(b,biome13_sample,pft_param_names,u_params,n_psamp,psample)
    biome10_sample = hmatch_biome(b, biome_name, sample, emulator_path, obs, minimize)
    if biome10_sample.isna().all().all():
        return pd.DataFrame({np.nan})
    
    ############
    # C tree
    b = 8 # 'Conifer forest'
    biome_name = biome_configs[b]['name']
    cols = np.concatenate([u_params,pft_param_names[13],pft_param_names[14]]).tolist()
    Aset = biome4_sample[cols]
    cols = np.concatenate([pft_param_names[2]]).tolist()
    Bset = biome10_sample[cols]
    merged = pd.concat([Aset.reset_index(drop=True), Bset.reset_index(drop=True)], axis=1)
    sample = create_biome_sample(b,merged,pft_param_names,u_params,n_psamp,psample)
    biome8_sample = hmatch_biome(b, biome_name, sample, emulator_path, obs, minimize)
    if biome8_sample.isna().all().all():
        return pd.DataFrame({np.nan})

    ####
    b = 7 # 'Mixed deciduous temperate forest'
    biome_name = biome_configs[b]['name']
    sample = create_biome_sample(b,biome8_sample,pft_param_names,u_params,n_psamp,psample)
    biome7_sample = hmatch_biome(b, biome_name, sample, emulator_path, obs, minimize)
    if biome7_sample.isna().all().all():
        return pd.DataFrame({np.nan})
    
    ####
    b = 11 # 'Broadleaf deciduous boreal trees' # 2,8,12,13
    biome_name = biome_configs[b]['name']
    cols = np.concatenate([u_params,pft_param_names[2],pft_param_names[13]]).tolist()
    b8 = biome8_sample[cols]
    cols = np.concatenate([pft_param_names[12]]).tolist()
    b10 = biome10_sample[cols]
    merged = pd.concat([b8.reset_index(drop=True), b10.reset_index(drop=True)], axis=1)
    sample = create_biome_sample(b,merged,pft_param_names,u_params,n_psamp,psample)
    biome11_sample = hmatch_biome(b, biome_name, sample, emulator_path, obs, minimize)
    if biome11_sample.isna().all().all():
        return pd.DataFrame({np.nan})

    b_samples = [0.0,biome1_sample, biome2_sample, biome3_sample, biome4_sample, biome5_sample, biome6_sample, 
             biome7_sample, biome8_sample, biome9_sample, biome10_sample, biome11_sample, biome12_sample, biome13_sample]

    return b_samples


