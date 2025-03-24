
for i in {0000..1500}; do
    lhc='lhc'$i
    job=$lhc".job"
    if grep -q $lhc good.txt; then 
        x=5 
    else 
        echo $lhc" does not exist"
        sed 's/key/'$lhc'/g' template.sh > $job
        qsub $job
    fi
    
done

