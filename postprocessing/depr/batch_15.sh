
for i in {0..99}; do
    echo $i
    job='batch'$i'.job'
    sed 's/key/'$i'/g' template.sh > $job
    qsub $job

done

