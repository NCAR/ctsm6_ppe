import numpy as np
import pandas as pd
import xarray as xr
import glob
import sys

def amean(da):
    #annual mean
    m  = da['time.daysinmonth']
    xa = 1/365*(m*da).groupby('time.year').sum().compute()
    xa.name=da.name
    xa.attrs=da.attrs
    return xa

def preprocess(ds):
    dvs=['TLAI','TWS','GPP','NPP','TOTVEGC','TOTSOMC','TOTECOSYSC']
    return ds[dvs]

lhc=sys.argv[1]

s='/glade/derecho/scratch/linnia/'
c='BNF_v2.n01_ctsm5.3.012_transient_postSASU_'+lhc
d=s+c+'/run/'
f=sorted(glob.glob(d+'*.h0.*'))

ds=xr.open_mfdataset(f,combine='nested',concat_dim='time',preprocess=preprocess)
ds['time']=xr.cftime_range('2000',periods=len(ds.time),freq='MS',calendar='noleap')
dsout=xr.Dataset({v:amean(ds[v]) for v in ds.data_vars})
nan=dsout.isel(year=0)*np.nan
nan['year']=1999
x=xr.concat([nan,dsout],dim='year')
x['time']=x.year
x=x.swap_dims({'year':'time'})


dout='/glade/work/linnia/CLM6-PPE/ctsm6_lhc/postp/tmp/'
fout=dout+f[0].split('/')[-1].split('clm2')[0]+'postp_pSASU.nc'
x.to_netcdf(fout)