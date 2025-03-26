
for i in {0..1}; do
    
    sed 's/num/'$i'/g' template.sh > 'Ctree_'$i'.job'
    qsub 'Ctree_'$i'.job'

done
