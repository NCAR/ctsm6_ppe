#!/bin/bash
# copy files from scratch to ddir

# ===============================================
# history files
ddir="/glade/campaign/cgd/tss/projects/PPE/ctsm530_OAAT/hist/"

cd $SCRATCH

#ens_list=`ls -d ctsm5.3.0_transient_oaat*`
ens_list=`ls -d ctsm5.3.0_transient_oaat019*`

for member in $ens_list
do
    echo $member
    cd ${member}/run/
    cp ctsm5.3.0_transient_oaat*.clm2.h* $ddir
    cd $SCRATCH

done

# ===============================================
# restart files
ddir="/glade/campaign/cgd/tss/projects/PPE/ctsm530_OAAT/rest/"

cd $SCRATCH

#ens_list=`ls -d ctsm5.3.0_transient_oaat*`
ens_list=`ls -d ctsm5.3.0_transient_oaat019*`

for member in $ens_list
do
    echo $member
    cd ${member}/run/
    cp ctsm5.3.0_transient_oaat*.clm2.r.1985-01-01-00000.nc $ddir
    cp ctsm5.3.0_transient_oaat*.clm2.r.2024-01-01-00000.nc $ddir
    cd $SCRATCH

done