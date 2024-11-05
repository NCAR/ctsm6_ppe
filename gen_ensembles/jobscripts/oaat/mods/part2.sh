#!/bin/bash
prevcase=$1 
CASENAME=$(basename $2)
scratch='/glade/derecho/scratch/linnia/'

# ########################################################################################
# PART 2
# ########################################################################################
#
#    This script adds daily history fields
#    and then the model will run for 1985-2000 as a branch run
#
# ########################################################################################


# --- Ensure that the env_run.xml file has the correct content
./xmlchange RUN_TYPE=branch
./xmlchange RUN_REFCASE=$CASENAME
./xmlchange RUN_REFDATE=1985-01-01
./xmlchange STOP_OPTION=nyears
./xmlchange STOP_N=16
./xmlchange CONTINUE_RUN=FALSE
./xmlchange RESUBMIT=0

# save original user_nl_clm
cp user_nl_clm user_nl_clm_transient_1901-1985

# Add in the daily history output items 
echo -e "\n">> user_nl_clm
cat user_nl_clm_daily_1985 >> user_nl_clm
