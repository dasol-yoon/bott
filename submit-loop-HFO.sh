#!/bin/sh

for trial in $(seq 1 1 1)
do
    for algo in EI EICF KG Random
    do
        for num_iter in 50
        do
            for n_init_evals in 7
            do
                sbatch -J HFO_${trial}_${algo} \
                    -o ./logs/HFO_Tri_${trial}_${algo}_%j.out \
                        -e ./logs/HFO_Tri_${trial}_${algo}_%j.err \
                        --requeue HFO.sub ${trial} ${algo} ${num_iter} ${n_init_evals}
                    sleep 0.1s
            done
        done
    done
done