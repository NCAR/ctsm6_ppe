
for i in {0000..0000}; do
    
    sed 's/num/'$i'/g' template.sh > 'pft_'$i'.job'
    qsub 'pft_'$i'.job'

done
