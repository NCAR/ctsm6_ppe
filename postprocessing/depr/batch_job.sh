
for i in {0002..0500}; do
    echo $i
    #sed 's/key/lhc'$i'/g' template.sh > 'postp_'$i'.job'
    sed 's/key/wave1'$i'/g' template.sh > 'postp_'$i'.job'
    qsub 'postp_'$i'.job'

done

