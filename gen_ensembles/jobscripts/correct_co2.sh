#!/bin/bash

#########################################################################################
#  Correcting CO2 for CLM6 LHC ensemble
#########################################################################################
#
#    This script changes the datastreams to recycle 2014 aerosols
#    and switches to TRENDY CO2 
#    and then runs the model for 2015-2023 as a hybrid run 
#
#########################################################################################

for ens in $(seq -w 1001 1500); do
    # Define the CASENAME variable with the current folder number
    CASENAME="/glade/work/linnia/BNF_v2.n01_ctsm5.3.012/cime/scripts/transient/BNF_v2.n01_ctsm5.3.012_transient/BNF_v2.n01_ctsm5.3.012_transient_lhc${ens}/"

    # Print the CASENAME for debugging (optional)
    echo "Running: $CASENAME"

    cd $CASENAME

    # --- Ensure that the env_run.xml file has the correct content
    ./xmlchange PROJECT=P93300041
    ./xmlchange RUN_TYPE=hybrid
    ./xmlchange RUN_REFDATE=2015-01-01
    ./xmlchange RUN_STARTDATE=2015-01-01
    ./xmlchange CONTINUE_RUN=FALSE
    
    ./xmlchange STOP_OPTION=nyears
    ./xmlchange STOP_N=9
    ./xmlchange RESUBMIT=0

    # --- Change datastreams to recycle 2014 annual areosols an update CO2 stream
    cp user_nl_clm user_nl_clm_1985-2014

    #comment out any finidat and use_init_interp from user_nl_clm
	:> user_nl_clm.tmp
	while read line; do
	    if [[ $line = *"finidat"* || $line = *"use_init_interp"* ]]; then
		echo "!$line">>user_nl_clm.tmp
	    else
		echo "$line">>user_nl_clm.tmp
	    fi
	    done<user_nl_clm
	mv user_nl_clm.tmp user_nl_clm

    cp /glade/u/home/linnia/ctsm6_ppe/gen_ensembles/jobscripts/user_mods/user_nl_datm_streams_CRUJRA.2015-2023 . 
    cp user_nl_datm_streams_CRUJRA.2015-2023 user_nl_datm_streams
    
    ./case.submit

    sleep 20

done
