#!/bin/bash
# copy files from scratch to campaign

# ===============================================

hist_dir="/glade/campaign/cgd/tss/projects/PPE/ctsm6_oaat/hist/" # history files
rest_dir="/glade/campaign/cgd/tss/projects/PPE/ctsm6_oaat/rest/" # restart files 

cd $SCRATCH

ens_list=`ls -d BNF_v2.n01_ctsm5.3.012_transient_oaat*`

for member in $ens_list
do
    echo $member
    cd ${member}/run/
    cp BNF_v2.n01_ctsm5.3.012_transient_oaat*.clm2.h* $hist_dir
    
    cp BNF_v2.n01_ctsm5.3.012_transient_oaat*.clm2.r.1985-01-01-00000.nc $rest_dir
    cp BNF_v2.n01_ctsm5.3.012_transient_oaat*.clm2.r.2024-01-01-00000.nc $rest_dir

    cd $SCRATCH

done
