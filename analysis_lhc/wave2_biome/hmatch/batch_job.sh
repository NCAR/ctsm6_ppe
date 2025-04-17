
for i in {500..999}; do
    
    sed 's/key/'$i'/g' template.sh > 'wave2_hmatch_'$i'.job'
    qsub 'wave2_hmatch_'$i'.job'

done
