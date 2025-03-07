import xarray as xr
import numpy as np
import glob
import sys
sys.path.append('/glade/u/home/linnia/ctsm6_ppe/')
from utils.pyfunctions import *


fifteen=int(sys.argv[1])
y1=2004
y2=2023

######################################################
# setup 
dir='/glade/campaign/cgd/tss/projects/PPE/ctsm6_lhc/hist/'
out_dir='/glade/derecho/scratch/djk2120/postp/ppe/ctsm6/'
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

lhcs=['lhc'+str(1+fifteen*15+i).zfill(4) for i in range(15)]
if fifteen==0:
    lhcs=['lhc0000',*lhcs]
    

######################################################
# load and process data

for lhc in lhcs:
    print(lhc)
    # all the files
    f=sorted(glob.glob(dir+'*'+lhc+'*.'+tape+'.*'))
    suff='postp_grid.nc'

    
    if len(f)==35:
        
        # finding only the needed files
        y1,y2=2004,2023
        years=np.array([f.split('.')[-2][:4] for f in f]).astype(int)
        ny=years[1]-years[0]
        y1s=years
        y2s=years+ny-1
        ixf=(y2s>=y1)&(y1s<=y2)
        files=np.array(f)[ixf]
        
        # subset and average
        ds=xr.open_mfdataset(files,combine='by_coords',preprocess=pp)
        out=xr.Dataset()
        for v in dvs:
            out[v+'_gridded_mean']=amean(ds[v]).sel(year=slice(y1,y2)).mean(dim='year')
            out[v+'_gridded_mean'].attrs =ds[v].attrs
        
        # save 
        fout=out_dir+f[0].split('/')[-1].split('clm2')[0]+suff
        out.attrs={'time_period':str(y1)+'-'+str(y2)}
        out.to_netcdf(fout)

    else:
        print('bad files... crash?')
        f=glob.glob(out_dir+'*lhc0000*'+suff)[0]
        ds=np.nan*xr.open_dataset(f)
        ds.attrs['crashed']='True'
        fout=f.replace('lhc0000',lhc)
        ds.to_netcdf(fout)
        
        
