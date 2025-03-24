import xarray as xr
import numpy as np
import glob
import sys
sys.path.append('/glade/u/home/linnia/ctsm6_ppe/')
from utils.pyfunctions import *
utils_path = '/glade/u/home/linnia/ctsm6_ppe/utils/'

ens=sys.argv[1] 

######################################################
# setup 
dir='/glade/campaign/cgd/tss/projects/PPE/ctsm6_lhc/hist/'
out_dir='/glade/work/linnia/CLM6-PPE/ctsm6_lhc/postp/tmp/'
tape='h1'

dvs = ['TOTVEGC','NPP','NPP_NUPTAKE','HTOP','GPP','TLAI','FCTR','FGEV','FCEV','BTRANMN','AR','AGNPP','pfts1d_itype_veg']

def pp(ds):
    yr0=str(ds['time.year'][0].values)
    nt=len(ds.time)
    ds['time'] = xr.cftime_range(yr0,periods=nt,freq='MS',calendar='noleap') #fix time bug
    return ds[dvs]

######################################################
# load and process data

f=sorted(glob.glob(dir+'*'+ens+'*.'+tape+'.*'))
ds=xr.open_mfdataset(f,combine='by_coords',preprocess=pp)

# calculate pft mean
yr1 = '1985'
yr2 = '2023'

out=xr.Dataset()
    
for v in dvs:
    if v=='TLAI':
        x=amax(ds[v].sel(time=slice(yr1,yr2)))
        out[v+'_biome_amax'] = pxbmean(x)
    else:
        x=amean(ds[v].sel(time=slice(yr1,yr2)))
        out[v+'_biome_amean'] = pxbmean(x)

    for dv in out.data_vars:
        if v in dv:
            out[dv].attrs=ds[v].attrs

# save 
fout=out_dir+f[0].split('/')[-1].split('clm2')[0]+'postp_pxbmean.nc'
out.to_netcdf(fout)

