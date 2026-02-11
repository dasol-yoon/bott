#!/bin/sh

for trial in $(seq 15 1 16)
do
    for algo in EICF 
    do
        for num_iter in 50
        do
            for param_truth in "380 1.5 -1.5" 
            do
                sbatch -J run0_${trial}_${algo} \
                       -o ./logs/${trial}_${algo}_%j.out \
                       -e ./logs/${trial}_${algo}_%j.err \
                       --requeue run_8.sub ${trial} ${algo} ${num_iter} "$param_truth"
                sleep 0.1s
            done
        done
    done
done