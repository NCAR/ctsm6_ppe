import xarray as xr
import numpy as np
import glob
import sys
sys.path.append('/glade/u/home/linnia/ctsm6_ppe/')
from utils.pyfunctions import *
utils_path = '/glade/u/home/linnia/ctsm6_ppe/utils/'

lhc=sys.argv[1]
in_dir=sys.argv[2]
out_dir=sys.argv[3]

if in_dir[-1]!='/':
    in_dir+='/'
if out_dir[-1]!='/':
    out_dir+='/'
######################################################
# setup
dvs=['NPP','GPP','AR','NPP_NUPTAKE','TLAI','BTRANMN','Vcmx25Z']
def pp(ds):
    yr0=str(ds['time.year'][0].values)
    nt=len(ds.time)
    ds['time'] = xr.cftime_range(yr0,periods=nt,freq='MS',calendar='noleap') #fix time bug
    return ds[dvs]
def bavg(da,la):
    if 'year' not in da.dims:
        raise ValueError('bavg meant for annualized data')
    x=(la*da).sum(dim=['pft','vegtype'])/la.sum(dim=['pft','vegtype'])
    return x
######################################################
# find files for biome and global analysis
files={}
for tape in ['h0','h1']:
    keys=[f.split('transient')[1].split('.')[0][1:] for f in sorted(glob.glob(in_dir+'*'+tape+'.2020-*'))]
    k0=keys[0]
    f0=sorted(glob.glob(in_dir+'*'+k0 +'*.'+tape+'.*'))
    f =sorted(glob.glob(in_dir+'*'+lhc+'*.'+tape+'.*'))
    if len(f)<len(f0):
        #hacky way to generate correctly shaped nan output for failed simulations
        # if the requested lhc is missing, we will reanalyze lhc0000 and multiply by np.nan
        bad=True
        f=f0
    else:
        bad=False
    files[tape]=f
fout=out_dir+f[0].split('/')[-1].split('clm2')[0]+'postp.nc'

#load h1 data, analyzing 2000-2023
ds=xr.open_mfdataset(files['h1'][-5:],combine='by_coords',preprocess=pp)
f=utils_path+'lapxb_sg_sudoku_ctsm53017.nc'
tmp=xr.open_dataset(f)
la=tmp.lapxb_sg

#precursors
gppg=bavg(amean(ds.GPP),la).mean(dim='year')
nppnupg=bavg(amean(ds.NPP_NUPTAKE),la).mean(dim='year')
ar=bavg(amean(ds.AR),la).mean(dim='year')
arwonit=ar-nppnupg
lai=bavg(amean(ds.TLAI),la).mean(dim='year')
nppg=bavg(amean(ds.NPP),la).mean(dim='year')

#biome-level metrics
vcmx25z=bavg(amean(ds.Vcmx25Z),la).mean(dim='year')
ncost=nppnupg/gppg
rcost=arwonit/gppg
wstress=bavg(amean(ds.GPP*(1-ds.BTRANMN)),la).mean(dim='year')/gppg
alloc=lai/nppg
cue=nppg/gppg

#load h0 data, analyzing 2000-2023
ds=xr.open_mfdataset(files['h0'][-5:],combine='by_coords',preprocess=pp)
la=xr.open_dataset(utils_path+'landarea_retrain_h0.nc').landarea

#precursors
gppg=gmean(amean(ds.GPP),la).mean(dim='year')
nppnupg=gmean(amean(ds.NPP_NUPTAKE),la).mean(dim='year')
ar=gmean(amean(ds.AR),la).mean(dim='year')
arwonit=ar-nppnupg
lai=gmean(amean(ds.TLAI),la).mean(dim='year')
nppg=gmean(amean(ds.NPP),la).mean(dim='year')

#global metrics
vcmx25zg=gmean(amean(ds.Vcmx25Z),la).mean(dim='year')
ncostg=nppnupg/gppg
rcostg=arwonit/gppg
wstressg=gmean(amean(ds.GPP*(1-ds.BTRANMN)),la).mean(dim='year')/gppg
allocg=lai/nppg
cueg=nppg/gppg

# concatenate and save to outbound dataset
out=xr.Dataset()
gvs=[vcmx25zg,ncostg,rcostg,wstressg,allocg,cueg]
bvs=[vcmx25z,ncost,rcost,wstress,alloc,cue]
dvs=['vcmx25z','ncost','rcost','wstress','alloc','cue']
for bv,gv,dv in zip(bvs,gvs,dvs):
    out[dv]=xr.concat([bv,xr.DataArray([gv],dims='biome')],dim='biome')

# metadata
out['vcmx25z'].attrs={'units':'umol/m2/s','long_name':'Vcmax predicted by LUNA'}
out['wstress'].attrs={'long_name':'Water stress','units':'-','note':'GPP(1-BT) at monthly, then accumulated ann and glob, then divided by GPP'}
out['alloc'].attrs={'long_name':'Leaf allocation efficiency','units':'m2/gC/s','note':'TLAI/NPP'}
out['cue'].attrs={'long_name':'Carbon use efficiency','units':'-','note':'NPP/(GPP)'}
out['ncost'].attrs={'long_name':'Carbon expenditure on nitrogen','units':'-','note':'NPP_NUPTAKE/(GPP)'}
out['rcost'].attrs={'long_name':'Carbon expenditure on other respiration','units':'-','note':'(AR-NPP_NUPTAKE)/(GPP)'}
out['biome_name']=[*tmp.biome_name.values,'Global']


# nan output if no files
if bad:
    out=np.nan*out
    fout=fout.replace(k0,lhc)
    
# save 
out.to_netcdf(fout)

