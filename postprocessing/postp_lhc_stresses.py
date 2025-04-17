import xarray as xr
import numpy as np
import glob
import sys
sys.path.append('/glade/u/home/linnia/ctsm6_ppe/')
from utils.pyfunctions import *
utils_path = '/glade/u/home/linnia/ctsm6_ppe/utils/'

lhc=sys.argv[1] 
out_dir=sys.argv[2]
if out_dir[-1]!='/':
    out_dir+='/'
######################################################
# setup 
dir='/glade/campaign/cgd/tss/projects/PPE/ctsm6_lhc/hist/'
tape='h0'
dvs=['GPP','NPP','NPP_NUPTAKE','TLAI','BTRANMN']

def pp(ds):
    yr0=str(ds['time.year'][0].values)
    nt=len(ds.time)
    ds['time'] = xr.cftime_range(yr0,periods=nt,freq='MS',calendar='noleap') #fix time bug
    return ds[dvs]

######################################################
# find files
f0=sorted(glob.glob(dir+'*lhc0000*.'+tape+'.*'))[-5:]
f=sorted(glob.glob(dir+'*'+lhc+'*.'+tape+'.*'))[-5:]
if len(f)<len(f0):
    #hacky way to generate correctly shaped nan output for failed simulations
    # if the requested lhc is missing, we will reanalyze lhc0000 and multiply by np.nan
    bad=True
    f=f0
else:
    bad=False
fout=out_dir+f[0].split('/')[-1].split('clm2')[0]+'postp.nc'

#load data
ds=xr.open_mfdataset(f,combine='by_coords',preprocess=pp)

# calculate various stress metrics
la=xr.open_dataset(utils_path+'landarea_retrain_h0.nc').landarea
out=xr.Dataset()
# photosynthetic capacity
cf=24*60*60*365*la.sum()*1e-9
gpp=cf*amean(ds.GPP)
maxgpp=gmean(gpp.max(dim='year'),la)
# respiration and nitrogen acquisition
nppg=gmean(cf*amean(ds.NPP).mean(dim='year'),la)
cue=nppg/gmean(gpp.mean(dim='year'),la)
# nitrogen costs
nppnupg=gmean(cf*amean(ds.NPP_NUPTAKE).mean(dim='year'),la)
ncost=nppnupg/(nppnupg+nppg)
# water stress
btran=gmean(amean(ds.BTRANMN).mean(dim='year'),la)
# mortality and allocation
lai=gmean(amean(ds.TLAI).mean(dim='year'),la)
alloc=lai/nppg
# metadata
maxgpp.attrs={'units':'PgC/yr','long_name':'Upper limit GPP','note':'found max at each gridcell, then globally averaged'}
cue.attrs={'long_name':'Carbon use efficiency','units':'-','note':'NPP/GPP'}
btran.attrs={'long_name':'BTRANMN','units':'-','note':'midday minimum'}
alloc.attrs={'long_name':'Leaf allocation efficiency','units':'m2/m2/PgC/yr','note':'TLAI/NPP'}
ncost.attrs={'long_name':'Carbon expenditure on nitrogen','units':'-','note':'NPP_NUPTAKE/(NPP+NPP_NUPTAKE)'}
# save to outbound dataset
out['maxgpp']=maxgpp
out['cue']=cue
out['btran']=btran
out['alloc']=alloc
out['ncost']=ncost

# nan output if no files
if bad:
    out=np.nan*out
    fout=fout.replace('lhc0000',lhc)
    
# save 
out.to_netcdf(fout)

