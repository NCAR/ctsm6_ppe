#!/bin/bash

#CASENAME=$1
CASENAME='/glade/work/linnia/BNF_v2.n01_ctsm5.3.012/cime/scripts/transient/BNF_v2.n01_ctsm5.3.012_transient/BNF_v2.n01_ctsm5.3.012_transient_test0002/'
scratch='/glade/derecho/scratch/linnia/'

#########################################################################################
#  Correcting CO2 for CLM6 LHC ensemble
#########################################################################################
#
#    This script changes the datastreams to recycle 2014 aerosols
#    and switches to TRENDY CO2 
#    and then runs the model for 2015-2023 as a hybrid run 
#
#########################################################################################


cd $CASENAME
pwd 

# --- Ensure that the env_run.xml file has the correct content
./xmlchange RUN_TYPE=hybrid
#./xmlchange RUN_REFCASE=$CASENAME
./xmlchange RUN_REFDATE=2015-01-01
./xmlchange STOP_OPTION=nyears
./xmlchange STOP_N=9
./xmlchange CONTINUE_RUN=FALSE
./xmlchange RESUBMIT=0

# --- Change datastreams to recycle 2014 annual areosols an update CO2 stream
cp /glade/u/home/linnia/ctsm6_ppe/gen_ensembles/jobscripts/user_mods/user_nl_datm_streams_CRUJRA.2015-2023 . 
cp user_nl_datm_streams_CRUJRA.2015-2023 user_nl_datm_streams

./case.submit
