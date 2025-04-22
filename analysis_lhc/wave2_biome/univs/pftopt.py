import os
import numpy as np
import xarray as xr
import pandas as pd
import glob
import tensorflow as tf
import sys

iparam=int(sys.argv[1])
ipft=int(sys.argv[2])

emulator_path = '/glade/u/home/linnia/ctsm6_ppe/analysis_lhc/wave2_biome/emulators_biome'
bpfts={'NaN': [],
 'Tropical rainforest': [4],
 'Tropical savanna': [4, 14],
 'Subtropical savanna': [4, 6, 14],
 'Broadleaf evergreen temperate tree': [5, 13, 14],
 'Grasslands': [13, 14],
 'Shrubland': [10, 13, 14],
 'Mixed deciduous temperate forest': [1, 7, 13, 14],
 'Conifer forest': [1, 2, 13, 14],
 'Siberian larch': [3, 11, 12],
 'Boreal forest': [2, 11, 12],
 'Broadleaf deciduous boreal trees': [2, 8, 12, 13],
 'Boreal shrubland': [12],
 'Tundra': [11, 12]}
biomes=[b for b in bpfts]
npft=[len(bpfts[b]) for b in bpfts]
metrics=['lai','gpp','biomass']

obs = xr.open_dataset('/glade/u/home/linnia/ctsm6_ppe/analysis_lhc/wave2_biome/wave2_obsStatistics_sudokuBiomes.nc')
ovs={'lai':'LAI','gpp':'GPP','biomass':'biomassC'}

# get normalized default settings for universal params
params_lhc = pd.read_csv('/glade/work/linnia/CLM6-PPE/ctsm6_lhc/ctsm6lhc_11262024.txt').drop(columns='member')

pft_params   = ['kmax','psi50','jmaxb0','slatop','lmr_intercept_atkin',
                'medlynslope','medlynintercept','froot_leaf','leafcn','leaf_long',
                'KCN','dleaf','r_mort','fsr_pft','xl']
pftix=np.array([p in pft_params for p in params_lhc.columns])
u_params = params_lhc.columns[~pftix]
f='/glade/u/home/linnia/ctsm6_ppe/analysis_lhc/wave2_biome/calibrate/default_parameter_settings.csv'
default_params = pd.read_csv(f)
default_params = default_params.set_index('name')
tmp = default_params.transpose()
default = tmp.loc[['default norm']]
univ = default[u_params].values
univ[0,list(u_params).index('jmaxb1')]=0.4
univ[0,list(u_params).index('theta_cj')]=0.7


upfts=[]
for b in bpfts:
    for p in bpfts[b]:
        if p not in upfts:
            upfts.append(p)
upfts=sorted(upfts)



npft=17
nparam=15
nx=391
tol=1
diag=[np.zeros([npft,nparam])]
for i in upfts:
    for j in range(nparam):
        a=np.zeros([npft,nparam])
        a[i,j]=1
        diag.append(a)
        diag.append(-a)
diag=np.array(diag)


bset=biomes[1:]
bset.remove('Broadleaf evergreen temperate tree')
bms=[b+' '+m for b in bset for m in metrics]


pft_defaults=[]
for pft in range(17):
    dpft=[]
    for p in pft_params:
        px=p+'_'+str(pft)
        if p in default:
            dpft.append(default[p].values[0])
        elif px in default:
            dpft.append(default[px].values[0])
        else:
            dpft.append(0.5)
    pft_defaults.append(dpft)
pft_defaults=np.array(pft_defaults)


x0=np.tile(pft_defaults.reshape([1,17,15]),[11,1,1])
inits=np.linspace(0,1,11)
for i,init in enumerate(inits):
    x0[i,ipft,iparam]=init


for i in range(500):
    x1=np.vstack([np.clip(x0+0.15*diag*np.random.rand(nx,npft,nparam),0,1) for x0 in x0])
    error=[]
    dsout=xr.Dataset()
    for b in bset:
        bidx=biomes.index(b)
        s=np.hstack([x1[:,p,:] for p in bpfts[b]])
        ns=s.shape[0]
        s=np.hstack([np.tile(univ,[ns,1]),s])
        
        for v in metrics:
            loaded_emulator = tf.saved_model.load(emulator_path + v + '/' + b)
            y,yp=loaded_emulator.predict(s)
            
            obs_mean=obs[ovs[v]+'_mean'][bidx].values
            obs_std =obs[ovs[v]+'_stdev'][bidx].values
            
            ev=abs(y.numpy()-obs_mean)/obs_std
            if b=='Boreal shrubland':
                if v=='biomass':
                    ev=0*ev #ignoring this combo
            error.append(ev)

    error=np.hstack(error)
    eclip=np.clip(error,tol,np.inf)
    bests=eclip.sum(axis=1).reshape([len(inits),nx]).argmin(axis=1)+np.arange(len(inits))*nx
    x0=x1[bests,:,:]

    dsout['solution']=xr.DataArray(x0,dims=['init','pft','param'])
    dsout['errors']=xr.DataArray(error[bests,:],dims=['init','metric'])
    dsout['metric']=bms
    dsout['param']=pft_params
    dsout['init']=inits
    dout='/glade/derecho/scratch/djk2120/pftopt/'
    fout=dout+'pft'+str(ipft).zfill(2)+'param'+str(iparam).zfill(2)+'_iter'+str(i).zfill(3)+'.nc'
    dsout.to_netcdf(fout)




