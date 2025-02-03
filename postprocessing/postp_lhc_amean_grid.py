import xarray as xr
import numpy as np
import glob
import sys
sys.path.append('/glade/u/home/linnia/ctsm6_ppe/')
from utils.pyfunctions import *
utils_path = '/glade/u/home/linnia/ctsm6_ppe/utils/'

lhc=sys.argv[1] 

######################################################
# setup 
dir='/glade/campaign/cgd/tss/projects/PPE/ctsm6_lhc/hist/'
out_dir='/glade/work/linnia/CLM6-PPE/ctsm6_lhc/postp/tmp/'
tape='h0'

dvs=['GPP','AR','HR','NPP','NBP','NEP','ER',
     'EFLX_LH_TOT','FCTR','FCEV','FGEV','BTRANMN','FGR','FSH',
     'SOILWATER_10CM','TWS','QRUNOFF','SNOWDP','H2OSNO','FSNO',
     'TLAI','FSR','ALTMAX','TV','TG','NPP_NUPTAKE','LAND_USE_FLUX',
     'FAREA_BURNED','COL_FIRE_CLOSS',
     'TOTVEGC','TOTECOSYSC','TOTSOMC_1m',
     'TOTVEGN','TOTECOSYSN']

def pp(ds):
    yr0=str(ds['time.year'][0].values)
    nt=len(ds.time)
    ds['time'] = xr.cftime_range(yr0,periods=nt,freq='MS',calendar='noleap') #fix time bug
    return ds[dvs]

######################################################
# load and process data

f=sorted(glob.glob(dir+'*'+lhc+'*.'+tape+'.*'))
ds=xr.open_mfdataset(f,combine='by_coords',preprocess=pp)

out=xr.Dataset()
    
for v in dvs:

        out[v+'_gridded_amean']=amean(ds[v])
        out[v+'_gridded_amean'].attrs =ds[v].attrs

# save 
fout=out_dir+f[0].split('/')[-1].split('clm2')[0]+'postp_grid.nc'
out.to_netcdf(fout)
