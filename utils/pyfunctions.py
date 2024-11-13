import os
import numpy as np
import xarray as xr
import cftime
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import glob
import dask

def amean(da):
    #annual mean of monthly data
    m  = da['time.daysinmonth']
    cf = 1/365
    xa = cf*(m*da).groupby('time.year').sum().compute()
    return xa

def pftmean(da,lapft,pft):
    x=1/lapft.groupby(pft).sum()*(lapft*da).groupby(pft).sum()
    return x.compute()
    
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
                appends[p]=xr.DataArray(np.concatenate(([np.nan],df[p].values)),dims='ens')
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
    fla=utils_path+'landarea_transient.nc'
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
    
    whit = xr.open_dataset(utils_path+'whit/whitkey.nc')
    ds['biome']=whit.biome
    ds['biome_name']=whit.biome_name
                
    #get the pft names
    pfts=xr.open_dataset('/glade/campaign/asp/djk2120/PPEn11/paramfiles/OAAT0000.nc').pftname
    pfts=[str(p)[2:-1].strip() for p in pfts.values][:17]
    ds['pft_name']=xr.DataArray(pfts,dims='pft_id')
    
    return ds

# =========================================================

def get_map(da,sgmap=None):
    if not sgmap:
        sgmap=xr.open_dataset('sgmap.nc')
    return da.sel(gridcell=sgmap.cclass).where(sgmap.notnan).compute()

