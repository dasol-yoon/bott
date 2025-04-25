#!/bin/sh

for trial in $(seq 1 1 5)
do
    for algo in EI KG EICF TS
    do
        for num_iter in 50
        do
            for param_truth in "230 7 -3" 
            do
                sbatch -J PB_${trial}_${algo} \
                       -o ./output/${trial}_${algo}_%j.out \
                       -e ./output/${trial}_${algo}_%j.err \
                       --requeue submit.sub ${trial} ${algo} ${num_iter} "$param_truth"
                sleep 0.1s
            done
        done
    done
done
