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

# ================== general functions =======================

def amean(da):
    #annual mean of monthly data
    m  = da['time.daysinmonth']
    cf = 1/365
    xa = cf*(m*da).groupby('time.year').sum().compute()
    return xa

def amax(da):
    #annual max
    m  = da['time.daysinmonth']
    xa = da.groupby('time.year').max().compute()
    return xa

def pftmean(da,lapft,pft):
    # pft mean for individual PFT
    x=1/lapft.groupby(pft).sum()*(lapft*da).groupby(pft).sum()
    return x.compute()

def pmean(da,la):
    # pft mean for all PFTs
    xp=(1/la.groupby('pft').sum()*(da*la).groupby('pft').sum()).compute()
    return xp

#def pxbmean(da,minarea=5e4):
#    f='/glade/u/home/linnia/ctsm6_ppe/utils/lapxb_sg_sudoku_ctsm53017.nc'
#    lapxb=xr.open_dataset(f).lapxb_sg
#    ix=(lapxb.mean(dim='year').sum(dim='pft')>minarea)
#    dapb = 1/lapxb.sum(dim='pft')*(lapxb.isel(pxb=ix)*da).sum(dim='pft')
#    return dapb

def pxbmean(da):
    f='/glade/u/home/linnia/ctsm6_ppe/utils/lapxb_sg_sudoku_ctsm53017.nc'
    lapxb=xr.open_dataset(f).lapxb_sg
    x=(lapxb*da).sum(dim=['pft','vegtype'])/(lapxb).sum(dim=['pft','vegtype'])
    #xm=x.mean(dim='year')
    return x
    
def bmean(da,la,b):
    x=1/la.groupby(b).sum()*(la*da).groupby(b).sum()
    return x.compute()

def gmean(da,la):
    if 'gridcell' in da.dims:
        dim='gridcell'
    else:
        dim=['lat','lon']
    x=(da*la).sum(dim=dim)/la.sum()
    return x.compute()

def fix_time(ds):
    yr0=str(ds['time.year'][0].values)
    nt=len(ds.time)
    ds['time'] = xr.cftime_range(yr0,periods=nt,freq='MS',calendar='noleap') #fix time bug
    return ds

def get_map(da,sgmap=None):
    if not sgmap:
        sgmap=xr.open_dataset(os.path.join(utils_dir,'sgmap_retrain_h0.nc'))
    return da.sel(gridcell=sgmap.cclass).where(sgmap.notnan).compute()

# ===================== load ensemble ==============================

def get_files(exp,dir,key,tape,yy,utils_path):
    
    df=pd.read_csv(key)  
    yr0,yr1=yy
    
    if exp=='oaat':
        keys=df.key.values
        appends={v:xr.DataArray(df[v].values,dims='ens') for v in ['key','param','minmax']}
    else:
        keys = df.member.values
        appends={}
        params=[]
        for p in df.keys():
            if p!='member':
                appends[p]=xr.DataArray(df[p].values,dims='ens')
                params.append(p)
        appends['params']=xr.DataArray(params,dims='param')
        appends['key']=xr.DataArray(keys,dims='ens')
    
    fs   = np.array(sorted(glob.glob(dir+'*'+tape+'*')))
    yrs  = np.array([int(f.split(tape)[1][1:5]) for f in fs])
    
    #bump back yr0, if needed
    uyrs=np.unique(yrs)
    yr0=uyrs[(uyrs/yr0)<=1][-1]
    
    #find index to subset files
    ix    = (yrs>=yr0)&(yrs<=yr1)
    fs    = fs[ix] 
    
    #organize files to match sequence of keys
    ny=len(np.unique(yrs[ix]))
    
    fkeys=np.array([f.split('transient_')[1].split('.')[0] for f in fs])
    
    if ny==1:
        files=[fs[fkeys==k][0] for k in keys]
        dims  = 'ens'
    else:
        files=[list(fs[fkeys==k]) for k in keys]
        dims  = ['ens','time']
    
    #add landarea information
    #fla=utils_path+'landarea_transient.nc'
    fla=utils_path+'landarea_retrain_h0.nc'
    la=xr.open_dataset(fla)
    appends['la']=la.landarea
    #if tape=='h1':
    #    appends['lapft']=la.landarea_pft
        
    return files,appends,dims


def get_ds(files,dims,dvs=[],appends={},singles=[]):
    if dvs:
        def preprocess(ds):
            return ds[dvs]
    else:
        def preprocess(ds):
            return ds

    ds = xr.open_mfdataset(files,combine='nested',concat_dim=dims,
                           parallel=True,
                           preprocess=preprocess)
    f=np.array(files).ravel()[0]
    htape=f.split('clm2')[1][1:3]

    #add extra variables
    tmp = xr.open_dataset(f)
    for v in tmp.data_vars:
        if 'time' not in tmp[v].dims: 
            if v not in ds:
                ds[v]=tmp[v]
    
    #fix up time dimension, swap pft
    if (htape=='h0')|(htape=='h1'):
        yr0=str(ds['time.year'][0].values)
        nt=len(ds.time)
        ds['time'] = xr.cftime_range(yr0,periods=nt,freq='MS',calendar='noleap') #fix time bug
    if (htape=='h1'):
        ds['pft']=ds['pfts1d_itype_veg']
        
    
    for append in appends:
        ds[append]=appends[append]
        
             
    return ds

def get_exp(exp,dir,key,dvs,tape,yy,utils_path):
    '''
    exp: 'SSP370','transient','CTL2010','C285','C867','AF1855','2095','NDEP'
    dvs:  e.g. ['TLAI']    or [] returns all available variables
    tape: 'h0','h1',etc.
    yy:   e.g. (2005,2014) or () returns all available years
    '''
    files,appends,dims=get_files(exp,dir,key,tape,yy,utils_path)

    ds=get_ds(files,dims,dvs=dvs,appends=appends)
    
    f,a,d=get_files(exp,dir,key,'h0',yy,utils_path)
    singles=['RAIN','SNOW','TSA','RH2M','FSDS','WIND','TBOT','QBOT','FLDS']
    tmp=get_ds(f[0],'time',dvs=singles)
    for s in singles:
        ds[s]=tmp[s]
    
    if len(yy)>0:  
        ds=ds.sel(time=slice(str(yy[0]),str(yy[1])))
    
    ds['PREC']=ds.RAIN+ds.SNOW
    
    t=ds.TSA-273.15
    rh=ds.RH2M/100
    es=0.61094*np.exp(17.625*t/(t+234.04))
    ds['VPD']=((1-rh)*es).compute()
    ds['VPD'].attrs={'long_name':'vapor pressure deficit','units':'kPa'}
    ds['VP']=(rh*es).compute()
    ds['VP'].attrs={'long_name':'vapor pressure','units':'kPa'}
    
    whit = xr.open_dataset(utils_path+'whit/whitkey_CRUJRA.nc')
    ds['biome']=whit.biome
    ds['biome_name']=whit.biome_name
                
    #get the pft names
    pfts=xr.open_dataset('/glade/campaign/asp/djk2120/PPEn11/paramfiles/OAAT0000.nc').pftname
    pfts=[str(p)[2:-1].strip() for p in pfts.values][:17]
    ds['pft_name']=xr.DataArray(pfts,dims='pft_id')
    
    return ds


# ===================== Emulation ==============================

def train_val_save(X_train,X_test,y_train,y_test,kernel,outfile=None,savedir=None):

        model = gpflow.models.GPR(data=(X_train, np.float64(y_train)), kernel=kernel, mean_function=None)
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(model.training_loss, model.trainable_variables, options=dict(maxiter=30))

        # plot validation
        y_pred, y_pred_var = model.predict_y(X_test)
        sd = y_pred_var.numpy().flatten()**0.5

        coef_deter = r2_score(y_test,y_pred.numpy())

        if (savedir):
            print('saving')
            num_params = np.shape(X_train)[1]
            model.predict = tf.function(model.predict_y, input_signature=[tf.TensorSpec(shape=[None, num_params], dtype=tf.float64)])
            tf.saved_model.save(model, savedir)

        if (outfile):
            plt.figure()
            plt.errorbar(y_test, y_pred.numpy().flatten(), yerr=2*sd, fmt="o")
            plt.text(0.02, 0.98, f'R² = {np.round(coef_deter, 2)}',fontsize=10,transform=plt.gca().transAxes,va='top',ha='left')
            plt.text(0.02, 0.93, f'Emulator stdev ≈ {np.round(np.mean(sd), 2)}',fontsize=10,transform=plt.gca().transAxes,va='top',ha='left')
            plt.plot([0,np.max(y_test)],[0,np.max(y_test)],linestyle='--',c='k')
            plt.xlabel('CLM')
            plt.ylabel('Emulated')
            plt.xlim([np.min(y_test)-.5,np.max(y_test)+.5])
            plt.ylim([np.min(y_test)-.5,np.max(y_test)+.5])
            plt.tight_layout()
            plt.savefig(outfile)
    
        return coef_deter, np.mean(sd)


def select_kernel(kernel_dict,X_train,X_test,y_train,y_test):
    stdev = []
    r2 = []
    for k in range(len(kernel_dict)):
        kernel = kernel_dict[k]
        cd, sd = train_val_save(X_train,X_test,y_train,y_test,kernel,outfile=None,savedir=None)
        stdev.append(sd)
        r2.append(cd)
      
    r2_norm = (r2 - np.min(r2)) / (np.max(r2) - np.min(r2))
    std_norm = 1 - (stdev - np.min(stdev)) / (np.max(stdev) - np.min(stdev))

    score = 0.8*r2_norm + 0.2*std_norm
    best_kernel = kernel_dict[np.argmax(score)]
    
    return best_kernel


def build_kernel_dict(num_params):
    kernel_noise = gpflow.kernels.White(variance=1e-3)
    kernel_matern32 = gpflow.kernels.Matern32(active_dims=range(num_params), variance=10, lengthscales = np.tile(10,num_params))
    kernel_matern52 = gpflow.kernels.Matern52(active_dims=range(num_params),variance=1,lengthscales=np.tile(1,num_params))
    kernel_bias = gpflow.kernels.Bias(active_dims = range(num_params))
    kernel_linear = gpflow.kernels.Linear(active_dims=range(num_params),variance=[1.]*num_params)
    kernel_poly = gpflow.kernels.Polynomial(active_dims = range(num_params),variance=[1.]*num_params)
    kernel_RBF = gpflow.kernels.RBF(active_dims = range(num_params), lengthscales=np.tile(1,num_params))
    
    kernel_dict = {
        0:kernel_RBF + kernel_linear + kernel_noise,
        1:kernel_RBF + kernel_linear + kernel_noise + kernel_bias,
        2:kernel_poly + kernel_linear + kernel_noise,
        3:kernel_RBF + kernel_linear + kernel_noise + kernel_bias + kernel_poly,
        4:kernel_matern32,
        5:kernel_matern32*kernel_linear+kernel_noise,
        6:kernel_linear*kernel_RBF+kernel_matern32 + kernel_noise
    }
    return kernel_dict

# ===================================================
# ===========    History Matching   =================

def calc_I(model_mean,model_var,obs_mean,obs_var):
        # implausibility score
        I = np.abs(obs_mean-model_mean) / np.sqrt(obs_var + model_var)
        return I

def sample_and_score(data, N, n, num_params):
    index = np.arange(data.shape[0]) 

    # take N random samples of size n
    draws_ix = np.array([np.random.choice(index, size=n, replace=False) for _ in range(N)])

    # Compute scores for each sampled subset
    nbins = 20 # for LHC
    L = np.zeros(N) 
    for i in range(N):
        s = data[draws_ix[i, :], :]
        L[i] = LHC_score(n, nbins, num_params, s)
    
    # Find the best subset (minimum score)
    min_score_ix = np.argmin(L)
    
    return draws_ix[min_score_ix, :]


def LHC_score(n,nbins,num_params,sample):
    # sample is np array with rows as ensemble members and columns as parameters
    # zero is a perfect Latin Hypercube
    Pb = n/nbins
    dim_count = []
    for di in range(num_params):
        data = sample[:,di]
        bin_count = []
        for bi in range(nbins):
            bin_width = 1/20
            bin_min = bi/nbins
            bin_max = bi/nbins+bin_width
            Ab = np.sum((data>bin_min)&(data<bin_max))
    
            bin_count.append(np.abs(Pb - Ab))
        dim_count.append(np.sum(bin_count))
    
    return np.sum(dim_count)



