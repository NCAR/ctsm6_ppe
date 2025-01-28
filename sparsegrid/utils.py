import xarray as xr
import numpy as np

def gmean(da,la):
    if 'gridcell' in da.dims:
        dim='gridcell'
    else:
        dim=['lat','lon']
    x=(da*la).sum(dim=dim)/la.sum()
    x.attrs=da.attrs
    return x.compute()

def amean(da):
    #annual mean of monthly data
    m  = da['time.daysinmonth']
    cf = 1/365
    xa = cf*(m*da).groupby('time.year').sum().compute()
    return xa

def fix_time(ds):
    yr0=str(ds['time.year'][0].values)
    nt=len(ds.time)
    ds['time'] = xr.cftime_range(yr0,periods=nt,freq='MS',calendar='noleap') #fix time bug
    return ds