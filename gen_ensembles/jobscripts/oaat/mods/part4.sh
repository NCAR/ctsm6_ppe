#!/bin/bash
prevcase=$1 
CASENAME=$(basename $2)
scratch='/glade/derecho/scratch/linnia/'

#########################################################################################
# PART 4
#########################################################################################
#
#    This script changes the datastreams to recycle 2014 aerosols
#    and then runs the model for 2015-2023 as a branch run to get daily and subdaily output
#
#########################################################################################

# --- Ensure that the env_run.xml file has the correct content
./xmlchange RUN_TYPE=branch
./xmlchange RUN_REFCASE=$CASENAME
./xmlchange RUN_REFDATE=2015-01-01
./xmlchange STOP_OPTION=nyears
./xmlchange STOP_N=9
./xmlchange CONTINUE_RUN=FALSE
./xmlchange RESUBMIT=0

# --- Save 1850-2014 datastreams
cp user_nl_datm_streams user_nl_datm_streams.1901-2014

# --- Change datastreams to recycle 2014 annual areosols
cp user_nl_datm_streams_CRUJRA.2015-2023 user_nl_datm_streams

