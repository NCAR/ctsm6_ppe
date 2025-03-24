import xarray as xr
import numpy as np
import glob
import sys
sys.path.append('/glade/u/home/linnia/ctsm6_ppe/')
from utils.pyfunctions import *
utils_path = '/glade/u/home/linnia/ctsm6_ppe/utils/'

lhc=sys.argv[1] 
out_dir=sys.argv[2]

######################################################
# setup 
dir='/glade/campaign/cgd/tss/projects/PPE/ctsm6_lhc/hist/'
tape='h0'
dvs=['GPP']

def pp(ds):
    yr0=str(ds['time.year'][0].values)
    nt=len(ds.time)
    ds['time'] = xr.cftime_range(yr0,periods=nt,freq='MS',calendar='noleap') #fix time bug
    return ds[dvs]

######################################################
# find files
f0=sorted(glob.glob(dir+'*lhc0000*.'+tape+'.*'))
f=sorted(glob.glob(dir+'*'+lhc+'*.'+tape+'.*'))
if len(f)<len(f0):
    #hacky way to generate correctly shaped nan output for failed simulations
    bad=True
    f=f0
else:
    bad=False
fout=out_dir+f[0].split('/')[-1].split('clm2')[0]+'postp.nc'

#load data
ds=pp(xr.open_dataset(f[-2]))

# calculate global annual mean
la=xr.open_dataset(utils_path+'landarea_retrain_h0.nc').landarea
out=xr.Dataset()
for v in dvs:
    x=amean(ds[v])
    out[v+'_global_amean']=gmean(x,la)
    out[v+'_global_amean'].attrs=ds[v].attrs

# nan output if no files
if bad:
    out=np.nan*out
    fout=fout.replace('lhc0000',lhc)
    
# save 
out.to_netcdf(fout)

