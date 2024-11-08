#!/bin/bash
joblist=$1
template=$2
prevcase="none"
W=0           #track how many jobs are waiting 
:>cases.tmp   #where to write the new joblist
while read -r line;
do
    args=(${line//,/ }) #parse csv line to array 
    jobname=${args[0]}
    thiscase=${args[1]}
    setup=${args[2]}
    status=${args[3]}

    if [ $status = "queued" ]; then
	#run setup script
	cd $thiscase
	if [ $setup != "none" ]; then
	    $setup $prevcase $thiscase
	fi
	
	#submit case, capturing jobid
	./case.submit --resubmit-immediate
	X=$(./xmlquery JOB_IDS)
	arrX=(${X//:/ })
	jobid=${arrX[-1]} #save the last jobid
	
	#update job status to submitted
	cd -
 	echo $jobname","$thiscase","$setup",submitted">>cases.tmp
	
    elif [ $status = "waiting" ] && [ $W -eq 0 ];then
	#prepare next job for queue
	qj=$jobname".job"
	cp $template $qj
	sed -i 's:jobname:'$jobname':g' $qj
	sed -i 's:joblist:'$joblist':g' $qj
	sed -i 's:jobid:'$jobid':g' $qj

	#only queue the first waiting job
	((W++))

	#update job status to queued
	echo $jobname","$thiscase","$setup",queued">>cases.tmp
    else
	# job status unchanged
	echo $line>>cases.tmp
    fi
    prevcase=$thiscase
done < $joblist

#update joblist with new statuses
mv cases.tmp $joblist 

#queue next job
if [ $W -gt 0 ];then 
    qsub $qj
fi
