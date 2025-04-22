
import numpy as np
import xarray as xr
import pandas as pd
import tensorflow as tf
import sys
from univs_pyfunctions import *


#read user input
ustr0=sys.argv[1]
fout=sys.argv[2]
if 'and' in ustr0:
    ustrs=ustr0.split('and')
else:
    ustrs=[ustr0]

#setup stuff
emulator_path = '/glade/u/home/linnia/ctsm6_ppe/analysis_lhc/wave2_biome/emulators_biome'
obs = xr.open_dataset('/glade/u/home/linnia/ctsm6_ppe/analysis_lhc/wave2_biome/wave2_obsStatistics_sudokuBiomes.nc')

#biome and pft details
bpfts=get_bpfts()
biomes=[b for b in bpfts]
metrics=['lai','gpp','biomass']
pft_params,u_params,default=get_defaults()

# edit universal parameters according to ustr
usample = default[u_params]
u=usample.values
for ustr in ustrs:
    if ustr!='udef':
        i,n=ustr.split('u')[1].split('_')
        u[0,int(i)]=float(n)

#emulate and score
nsamp=10000
nrep=10
preds=make_preds(nsamp,nrep,usample,biomes,metrics,emulator_path,bpfts)
implaus=calc_imp(preds,biomes,obs)

#write to file
dicts=[implaus,preds['pred'],preds['var']]
names=['implausibility','pred','pred_var']
dsout=xr.Dataset({n:make_da(x) for x,n in zip(dicts,names)})
dsout['biome']=[b for b in implaus]
dsout['metric']=[b for b in implaus['Tundra']]
dsout['usamp']=[ustr0]

dout='/glade/derecho/scratch/djk2120/univ/'
dsout.to_netcdf(dout+fout)








