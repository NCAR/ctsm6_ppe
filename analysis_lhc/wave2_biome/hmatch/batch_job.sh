
for i in {4..100}; do
    
    sed 's/num/'$i'/g' template.sh > 'Ctree_'$i'.job'
    qsub 'Ctree_'$i'.job'

done
