#!/bin/sh

for trial in $(seq 2 1 20)
do
    for algo in EICF EI KG Random
    do
        for num_iter in 50
        do
            for n_init_evals in 7
            do
                sbatch -J STO28_${trial}_${algo} \
                    -o ./logs/STO28_Tri_${trial}_${algo}_%j.out \
                        -e ./logs/STO28_Tri_${trial}_${algo}_%j.err \
                        --requeue STO28.sub ${trial} ${algo} ${num_iter} ${n_init_evals}
                    sleep 0.1s
            done
        done
    done
done