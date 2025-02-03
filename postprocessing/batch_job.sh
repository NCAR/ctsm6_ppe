
for i in {0000..1500}; do
    echo $i
    sed 's/key/lhc'$i'/g' template.sh > 'lhc_'$i'.job'
    #sed 's/key/LHC'$i'/g' template.sh > 'lhc_'$i'.job'
    qsub 'lhc_'$i'.job'

done

