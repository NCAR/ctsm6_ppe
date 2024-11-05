#!/bin/bash
prevcase=$1 
CASENAME=$(basename $2)
scratch='/glade/derecho/scratch/linnia/'
##################################################################
# PART 1
#########################################################################################
#
#    This script sets up transient portion of the simulation (no longer recycling climate)
#    1901-1985
#
#########################################################################################

WDIR=$scratch$CASENAME'/run/'
DDIR=$WDIR'restart_dump/'
mkdir -p $DDIR

mv $WDIR$CASENAME.datm.rs1*.bin $DDIR
gzip $DDIR$CASENAME*.bin

./xmlchange STOP_OPTION=nyears
./xmlchange STOP_N=84
./xmlchange DATM_YR_ALIGN=1901
./xmlchange DATM_YR_START=1901
./xmlchange DATM_YR_END=2023
./xmlchange CONTINUE_RUN=TRUE

