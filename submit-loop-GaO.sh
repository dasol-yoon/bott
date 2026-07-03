#!/bin/sh

for trial in $(seq 6 1 6)
do
    for algo in EICF
    do
        for num_iter in 50
        do
            for n_init_evals in 7
            do
                sbatch -J GaO_${trial}_${algo} \
                    -o ./logs/GaO_Tri_${trial}_${algo}_%j.out \
                        -e ./logs/GaO_Tri_${trial}_${algo}_%j.err \
                        --requeue GaO.sub ${trial} ${algo} ${num_iter} ${n_init_evals}
                    sleep 0.1s
            done
        done
    done
done