pyscript="postp_lhc_stresses.py"
indir="/glade/campaign/cgd/tss/projects/PPE/ctsm6_wave2/hist/"
outdir="/glade/derecho/scratch/djk2120/postp/"
outfile="ctsm6lhc_global_stressors_wave2_2000-2023.nc"
author="djk2120@ucar.edu" #no whitespace!!
template="template.sh" #may need to edit the project code within

#prepare dirs
printf -v date '%(%Y%m%d)T' -1
out=$outdir"c"$date"/"
mkdir -p $out
mkdir -p $out"/concat/"
rm $out*.nc
rm *.job

#prepare job scripts (batches of 10 lhcs per job)
ct=0
i=0
nf=0
while (( $ct < 500 )); do
    ((ct++))
    ((nf++))
    printf -v lhc "%04d" $ct
    key="wave2"$lhc

    if (( $ct % 10 == 1 )); then
        ((i++))
        printf -v j "%03d" $i
        job="job"$j".job"
        sed 's/key/'$key'/g' $template > $job
    fi

    echo "python "$pyscript" "$key" "$indir" "$out >> $job

done

#submit job scripts
for job in *.job; do
    qsub $job
done

#babysit outdir and then concatenate
module load nco
s=0
while (( $s < 720 )); do
    ((s++))
    ncs=`ls -1 $out*.nc 2>/dev/null | wc -l`

    echo "s="$s", ncs="$ncs
    if (( $ncs < $nf )); then
        #waiting for enough files to appear
        sleep 60
    else
        echo "all done, now concatenating"
        s=1000

        #concatenate
        cfile=$out"concat/"$outfile
        if [ -f $cfile ]; then
            rm $cfile
        fi
        ncecat -u ens -h $out*.nc $cfile
        
        #annotate slightly
        ncatted -O -a script,global,a,c,$(pwd)"/"$pyscript -h $cfile
        ncatted -O -a author,global,a,c,$author -h $cfile
        ncatted -O -a date,global,a,c,$date -h $cfile
    fi
done

