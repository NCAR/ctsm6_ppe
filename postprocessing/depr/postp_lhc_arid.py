import xarray as xr
import numpy as np
import glob
import sys
sys.path.append('/glade/u/home/linnia/ctsm6_ppe/')
from utils.pyfunctions import *
utils_path = '/glade/u/home/linnia/ctsm6_ppe/utils/'

fifteen=int(sys.argv[1])

######################################################
# setup 
dir='/glade/campaign/cgd/tss/projects/PPE/ctsm6_lhc/hist/'
out_dir='/glade/derecho/scratch/djk2120/postp/ppe/arid/'
tape='h0'

dvs=['GPP','AR','HR','NPP','NBP','NEP','ER',
     'EFLX_LH_TOT','FCTR','FCEV','FGEV','BTRANMN','FGR','FSH',
     'SOILWATER_10CM','TWS','QRUNOFF',
     'TLAI','FSR','TV','TG','NPP_NUPTAKE','LAND_USE_FLUX',
     'FAREA_BURNED','COL_FIRE_CLOSS','RH2M','TSA']

lhcs=['lhc'+str(1+fifteen*15+i).zfill(4) for i in range(15)]
if fifteen==0:
    lhcs=['lhc0000',*lhcs]

def fix_time(ds):
    yr0=str(ds['time.year'][0].values)
    nt=len(ds.time)
    ds['time'] = xr.cftime_range(yr0,periods=nt,freq='MS',calendar='noleap') #fix time bug
    return ds

######################################################
# load and process data
lafile='/glade/u/home/djk2120/vp/lasg6_arid.nc'
la=xr.open_dataset(lafile).landarea

for lhc in lhcs:
    print(lhc)

    f=sorted(glob.glob(dir+'*'+lhc+'*.'+tape+'.*'))
    suff='postp_arid.nc'
    if len(f)==35:
        outs=[]
        for file in f:
            ds=fix_time(xr.open_dataset(file))
            out=xr.Dataset()
            for v in dvs:
                x=amean(ds[v])
                out[v+'_arid_amean']=gmean(x,la)
                out[v+'_arid_amean'].attrs=ds[v].attrs
            outs.append(out)
        out=xr.concat(outs,dim='year')
        
        # save 
        fout=out_dir+f[0].split('/')[-1].split('clm2')[0]+suff
        out.to_netcdf(fout)

    else:
        print('bad files... crash?')
        f=glob.glob(out_dir+'*lhc0000*'+suff)[0]
        ds=np.nan*xr.open_dataset(f)
        ds.attrs['crashed']='True'
        fout=f.replace('lhc0000',lhc)
        ds.to_netcdf(fout)

